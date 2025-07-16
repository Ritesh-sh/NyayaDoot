from fastapi import FastAPI, HTTPException, Query, APIRouter
from pydantic import BaseModel
from typing import List, Dict, Optional
import faiss
import pickle
import numpy as np
import torch
import re
import html
import time
from sentence_transformers import SentenceTransformer
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from fastapi.middleware.cors import CORSMiddleware
import together  # Updated import for new together API
import uvicorn
import os
import uuid
from dotenv import load_dotenv


load_dotenv()


TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
if not TOGETHER_API_KEY:
    print("⚠️ Warning: TOGETHER_API_KEY environment variable not set.")
    print("Please set this key to use Together AI models.")
    print("You can get a free API key at https://www.together.ai/")
together.api_key = TOGETHER_API_KEY  # Set API key for new together API
print("✅ Together AI SDK initialized successfully!")

# Load models
print("🔍 Loading models...")
model = SentenceTransformer("models/legal_embedding_model")

# Load FAISS index and section data
print("📂 Loading FAISS index...")
index = faiss.read_index("models/legal_index.faiss")

print("📚 Loading section data...")
with open("models/legal_sections.pkl", "rb") as f:
    data = pickle.load(f)
    section_data = data['section_data']
    all_acts = data['all_acts']

print("✅ All models loaded successfully!")

class ConversationState:
    def __init__(self):
        self.sessions = {}
        
    def get_session(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'current_context': None,
                'references': None,
                'cases': None,
                'history': []
            }
        return self.sessions[session_id]

    def update(self, session_id: str, query: str, answer: str, references: List[Dict], cases: List[Dict]):
        session = self.get_session(session_id)
        session['current_context'] = {
            'query': query,
            'answer': answer,
            'references': references,
            'cases': cases
        }
        session['references'] = references
        session['cases'] = cases
        session['history'].append(('user', query))
        session['history'].append(('assistant', answer))
        
    def get_conversation_history(self, session_id: str, max_turns: int =9) -> str:
        """Get formatted conversation history for context."""
        session = self.get_session(session_id)
        history = session['history']
        
        # Get the most recent turns (limited by max_turns)
        recent_history = history[-max_turns*2:] if history else []
        
        formatted_history = ""
        for i in range(0, len(recent_history), 2):
            if i+1 < len(recent_history):
                user_msg = recent_history[i][1]
                assistant_msg = recent_history[i+1][1]
                
                # Truncate very long messages to prevent context overflow
                if len(assistant_msg) > 500:
                    assistant_msg = assistant_msg[:500] + "..."
                
                formatted_history += f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"
        
        return formatted_history.strip()

conv_state = ConversationState()

def sanitize_query(query: str) -> str:
    query = html.escape(query.strip())
    return re.sub(r'\s+', ' ', query)

def extract_keywords_from_conversation(conversation_history: str, query: str) -> str:
    """Extract keywords with a focus on legal relevance"""
    # Combine conversation and query
    all_text = f"{conversation_history} {query}".lower()
    
    # Define common stopwords to filter out
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as', 'of',
                'can', 'could', 'would', 'should', 'will', 'shall', 'may', 'might',
                'must', 'have', 'has', 'had', 'do', 'does', 'did', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'i', 'me', 'my', 'mine', 'myself',
                'you', 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself',
                'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'we', 'us',
                'our', 'ours', 'ourselves', 'they', 'them', 'their', 'theirs', 'themselves',
                'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were',
                'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
                'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
                'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
                'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
                'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn',
                'wasn', 'weren', 'won', 'wouldn', 'could', 'would', 'shall',
                'should', 'will', 'may', 'might', 'must', 'ought', 'able'}
    
    # Split the text into words and filter out stopwords
    words = all_text.split()
    filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
    
    # Count word frequencies
    from collections import Counter
    word_counts = Counter(filtered_words)
    
    # Get common words as keywords
    common_keywords = [word for word, count in word_counts.most_common(10) if count > 1]
    
    # Add legal-specific terms if found in the text
    legal_terms = ['law', 'legal', 'right', 'duty', 'obligation', 'liability', 'contract', 
                  'agreement', 'court', 'judge', 'judgment', 'case', 'precedent', 'statute',
                  'act', 'section', 'clause', 'provision', 'regulation', 'rule']
    
    legal_keywords = [term for term in legal_terms if term in all_text]
    
    # Combine all keywords
    all_keywords = legal_keywords + common_keywords
    

    unique_keywords = list(set(all_keywords))
    
   
    if not unique_keywords:
        return "legal rights obligations duties"
    
    return ", ".join(unique_keywords[:15])  


def find_relevant_sections(query: str, conversation_history: str = "") -> List[Dict]:
    """Find legal sections relevant to the query using simple semantic search"""
    try:
        
        search_query = query
        if conversation_history:
            
            search_query = f"{conversation_history} {query}"
        
        # Get results using embedding search
        query_embedding = model.encode([search_query])
        D, I = index.search(query_embedding, 3)  # Get top 3 results
        
        # Process results
        results = []
        for i, idx in enumerate(I[0]):
            section = section_data[idx]
            
            # Generate explanation using Together AI
            prompt = f"""
            Explain in one sentence how this legal section is relevant to the user's query.
            Be specific about the connection between the section and the query.

            User's Query: {search_query}
            Section Text: {section['full_text'][:250]}  # Limit text length for token efficiency

            Provide a clear, concise explanation of the relevance.
            """
            
            try:
                relevance_explanation = generate_with_together(prompt)
                # Clean up the explanation
                relevance_explanation = relevance_explanation.strip()
                if relevance_explanation.startswith('"') and relevance_explanation.endswith('"'):
                    relevance_explanation = relevance_explanation[1:-1]
            except Exception as e:
                # Fallback explanation if AI generation fails
                relevance_explanation = f"This section contains provisions related to {query.split()[0]}"
            
            results.append({
                'act': section['act'],
                'section_number': section['section_number'],
                'full_text': section['full_text'],
                'summary': relevance_explanation,
                'score': float(D[0][i])  # Convert to float for JSON serialization
            })
                
        return results
            
    except Exception as e:
        return []


def fetch_kanoon_results(query: str, conversation_history: str = "") -> List[Dict]:
    """Fetch case law results from Indian Kanoon using general legal search"""
    search_query = query
    
    # If we have conversation history, use it to enhance the query
    if conversation_history:
        enhanced_query = extract_keywords_from_conversation(conversation_history, query)
        if enhanced_query and len(enhanced_query) > len(query)/2:
            search_query = enhanced_query
    
    # If query is very short, enhance it with general legal terms
    if len(search_query.split()) < 3:
        search_query += " legal rights precedent case law"
    
    print(f"Case search query: {search_query}")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = None
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get("https://indiankanoon.org/")
        search_box = driver.find_element(By.ID, "search-box")
        search_box.send_keys(search_query + Keys.RETURN)
        time.sleep(3)
        
        case_results = []
        case_elements = driver.find_elements(By.CSS_SELECTOR, "div.result_title > a")
        
        for i, element in enumerate(case_elements[:5]):  # Get top 5 results
            title = element.text.strip()
            url = element.get_attribute("href")
            
            # Try to get a snippet of the case content
            snippet = ""
            try:
                snippet_element = element.find_element(By.XPATH, "../../div[@class='snippet']")
                snippet = snippet_element.text.strip()
            except:
                # If snippet not available, just use the title
                snippet = title
                
            case_results.append({
                "title": title[:80],  # Truncate long titles
                "url": url,
                "snippet": snippet[:250] if snippet else ""  # Add a longer snippet
            })
            
        # If we didn't get any results, try a more general search
        if not case_results:
            print("No results found, trying a more general search...")
            driver.get("https://indiankanoon.org/")
            search_box = driver.find_element(By.ID, "search-box")
            
            # Use general legal terms for fallback search
            fallback_query = "legal rights precedent Supreme Court"
            search_box.send_keys(fallback_query + Keys.RETURN)
            time.sleep(3)
            
            case_elements = driver.find_elements(By.CSS_SELECTOR, "div.result_title > a")
            
            for i, element in enumerate(case_elements[:5]):  # Get top 5 results
                title = element.text.strip()
                url = element.get_attribute("href")
                
                # Try to get a snippet of the case content
                snippet = ""
                try:
                    snippet_element = element.find_element(By.XPATH, "../../div[@class='snippet']")
                    snippet = snippet_element.text.strip()
                except:
                    # If snippet not available, just use the title
                    snippet = title
                    
                case_results.append({
                    "title": title[:80],  # Truncate long titles
                    "url": url,
                    "snippet": snippet[:250] if snippet else ""
                })
            
        return case_results
        
    except Exception as e:
        print(f"Kanoon error: {e}")
        return []
    finally:
        if driver: driver.quit()

def fetch_specific_case_from_kanoon(case_name: str) -> Dict:
    """Search for a specific case name on Indian Kanoon and return the most relevant result"""
    print(f"Searching for specific case: {case_name}")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = None
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get("https://indiankanoon.org/")
        
        # Search with the exact case name for better precision
        search_box = driver.find_element(By.ID, "search-box")
        # Use quotes to search for exact phrase
        search_query = f'"{case_name}"'
        search_box.send_keys(search_query + Keys.RETURN)
        time.sleep(3)
        
        # Try to get the first (most relevant) result
        case_elements = driver.find_elements(By.CSS_SELECTOR, "div.result_title > a")
        
        if case_elements:
            # Get the first result
            element = case_elements[0]
            title = element.text.strip()
            url = element.get_attribute("href")
            
            # Try to get a snippet of the case content
            snippet = ""
            try:
                snippet_element = element.find_element(By.XPATH, "../../div[@class='snippet']")
                snippet = snippet_element.text.strip()
            except:
                # If snippet not available, just use the title
                snippet = title
                
            return {
                "title": title[:80],  # Truncate long titles
                "url": url,
                "snippet": snippet[:250] if snippet else "",  # Add a longer snippet
                "case_name": case_name  # Keep track of the original case name
            }
        else:
            # Try without quotes if no results found
            driver.get("https://indiankanoon.org/")
            search_box = driver.find_element(By.ID, "search-box")
            search_box.send_keys(case_name + Keys.RETURN)
            time.sleep(3)
            
            case_elements = driver.find_elements(By.CSS_SELECTOR, "div.result_title > a")
            
            if case_elements:
                element = case_elements[0]
                title = element.text.strip()
                url = element.get_attribute("href")
                
                # Try to get a snippet
                snippet = ""
                try:
                    snippet_element = element.find_element(By.XPATH, "../../div[@class='snippet']")
                    snippet = snippet_element.text.strip()
                except:
                    snippet = title
                    
                return {
                    "title": title[:80],
                    "url": url,
                    "snippet": snippet[:250] if snippet else "",
                    "case_name": case_name
                }
                
        # If we still didn't find anything, return a placeholder
        return {
            "title": f"Case: {case_name}",
            "url": f"https://indiankanoon.org/search/?formInput={case_name.replace(' ', '+')}",
            "snippet": f"This case was mentioned in the legal analysis but couldn't be found directly on Indian Kanoon.",
            "case_name": case_name
        }
        
    except Exception as e:
        print(f"Error searching for specific case: {e}")
        # Return a placeholder if there's an error
        return {
            "title": f"Case: {case_name}",
            "url": f"https://indiankanoon.org/search/?formInput={case_name.replace(' ', '+')}",
            "snippet": "Could not retrieve case details due to technical issues.",
            "case_name": case_name
        }
    finally:
        if driver: driver.quit()

def extract_case_names_from_text(text: str) -> List[str]:
    """Extract case names from text generated by the API"""
    # Common patterns for case citations
    case_patterns = [
        r'([A-Z][a-zA-Z\s]+)\s+vs\.?\s+([A-Z][a-zA-Z\s]+)\s+\((\d{4})\)',  # Pattern with year in parentheses
        r'([A-Z][a-zA-Z\s]+)\s+vs\.?\s+([A-Z][a-zA-Z\s]+)',                # Simple vs pattern
        r'([A-Z][a-zA-Z\s&]+)\s+v\.?\s+([A-Z][a-zA-Z\s&]+)',              # Alternative "v." pattern
    ]
    
    found_cases = []
    
    # Try each pattern
    for pattern in case_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                if len(match) == 3:  # Pattern with year
                    case_name = f"{match[0].strip()} vs. {match[1].strip()} ({match[2]})"
                else:  # Pattern without year
                    case_name = f"{match[0].strip()} vs. {match[1].strip()}"
                found_cases.append(case_name)
    
    # If no structured cases found, look for sections that might contain case names
    if not found_cases:
        # Look for lines mentioning cases
        for line in text.split('\n'):
            if any(indicator in line.lower() for indicator in ['case', 'precedent', 'judgement', 'supreme court', 'high court']):
                # Check if line has capitalized words that might be case names
                words = line.split()
                capitalized_words = [w for w in words if w and w[0].isupper()]
                if len(capitalized_words) >= 3:  # Enough capitalized words to potentially be a case name
                    found_cases.append(line.strip())
    
    # Return unique cases, limited to 3
    unique_cases = list(set(found_cases))
    return unique_cases[:3]

def fetch_cases_from_api_suggestions(api_response: str) -> List[Dict]:
    """Extract case names from API response and fetch them from Indian Kanoon"""
    # Get case names from the text
    case_names = extract_case_names_from_text(api_response)
    print(f"Extracted case names: {case_names}")
    
    if not case_names:
        # If no cases found, fall back to regular keyword search
        print("No specific cases found in API response, using standard search")
        return fetch_kanoon_results(api_response)
    
    # Fetch each case specifically
    results = []
    for case_name in case_names:
        case_result = fetch_specific_case_from_kanoon(case_name)
        if case_result:
            results.append(case_result)
            
    # If we couldn't find enough cases, supplement with regular search
    if len(results) < 2:
        # Extract keywords from the API response
        keywords = extract_keywords_from_conversation("", api_response)
        additional_results = fetch_kanoon_results(keywords)
        
        # Add only the results we don't already have
        existing_urls = [r.get('url', '') for r in results]
        for result in additional_results:
            if result.get('url') not in existing_urls:
                results.append(result)
                if len(results) >= 3:  # Limit to 3 total results
                    break
    
    return results[:3]  # Return up to 3 results

def generate_with_together(prompt: str) -> str:
    """Generate text with fallback mechanisms if API fails (updated for new together API)"""
    try:
        # More aggressive token optimization
        if len(prompt) > 1000:  # Reduced from 4000
            # For multiline prompts, trim more aggressively
            prompt_lines = prompt.split('\n')
            if len(prompt_lines) > 20:
                # Keep intro and conclusion but trim the middle
                prompt = '\n'.join(prompt_lines[:10] + 
                                  ['\n...[content trimmed]...\n'] + 
                                  prompt_lines[-10:])
            # Alternatively, trim very long text
            elif len(prompt) > 6000:
                prompt = prompt[:1500] + "\n...[content trimmed]...\n" + prompt[-1500:]
        print(f"Prompt length: {len(prompt)} characters")
        # Try Together AI with multiple models
        # Use new together API for chat completion
        response = together.ChatCompletion.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=256
        )
        # Extract and return the response
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm having trouble connecting to my knowledge source. For legal matters, it's always best to consult with a qualified attorney who can provide personalized advice."

def is_legal_query_together(query: str) -> bool:
    """
    Uses Together AI to determine if the query is legal in nature.
    Returns True if legal, False otherwise.
    (Updated for new together API)
    """
    prompt = f"""You are a classifier. Decide if the following user query is a legal question (about laws, rights, legal procedures, court cases, contracts, etc). \
Reply with only 'LEGAL' or 'NOT LEGAL'.\n\nQuery: \"{query}\"\n"""
    try:
        response = together.ChatCompletion.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5
        )
        result = response["choices"][0]["message"]["content"].strip().upper()
        return result == "LEGAL"
    except Exception as e:
        print(f"Error in legal query classification: {e}")
        # Fallback: assume legal (or you can default to False)
        return True


def generate_direct_answer(query: str, context: str = "", conversation_history: str = "", is_followup: bool = False) -> str:
    """Generate a direct answer with simplified context to reduce tokens"""
    # Only use conversation history if it's a follow-up query
    if not is_followup:
        conversation_history = ""
    
    # Limit conversation history to reduce tokens
    if len(conversation_history) > 500:
        # Extract just the last exchange or two
        parts = conversation_history.split("\n\n")
        if len(parts) > 2:
            conversation_history = "\n\n".join(parts[-2:])
    
    # Limit context length
    if len(context) > 1000:
        context = context[:1000] + "... [additional context omitted]"
    
    # Include conversation history for context-aware responses only if it's a follow-up
    history_context = f"\nPrevious conversation summary:\n{conversation_history}\n" if is_followup and conversation_history else ""
    
    prompt = f"""
    As a legal expert, provide 3 immediate steps for this situation:
    '{query}'
    {history_context}
    {f"Context: {context}" if context else ""}
    
    Format your response as 3 clear, actionable steps the person should take.
    Keep each step brief and practical.
    """
    answer = generate_with_together(prompt)
    return answer if answer else "Immediate steps:\n1. Consult with a qualified lawyer\n2. Document all relevant information\n3. Keep records of all communications"

def generate_legal_analysis(text: str, act_name: str, section_number: str) -> Dict:
    """Generate a concise and relevant summary of a legal section"""
    # Create a descriptive analysis based on the section text and act name
    act_name_clean = act_name.replace('.pdf', '').replace('-', ' ').title()
    text_lower = text.lower()
    
    # Extract key phrases from the text for a more specific description
    keyword_indicators = ["shall", "may", "means", "includes", "provided that", "deemed to be", "notwithstanding"]
    key_phrases = []
    
    # Look for the most relevant phrase
    for indicator in keyword_indicators:
        if indicator in text_lower:
            # Get the text around the indicator
            parts = text_lower.split(indicator)
            if len(parts) > 1:
                phrase = parts[1].split(".")[0]
                if len(phrase) < 50:  # Only use shorter phrases for conciseness
                    key_phrases.append((indicator, phrase))
    
    if key_phrases:
        # Use the most relevant phrase
        indicator, key_phrase = key_phrases[0]
        key_phrase = key_phrase[:50]  # Limit to 50 chars for brevity
        
        # Create a focused summary
        if "shall" in indicator:
            summary = f"Mandates that {key_phrase.strip()}"
        elif "may" in indicator:
            summary = f"Allows for {key_phrase.strip()}"
        elif "means" in indicator or "includes" in indicator:
            summary = f"Defines {key_phrase.strip()}"
        elif "provided that" in indicator:
            summary = f"Subject to condition that {key_phrase.strip()}"
        else:
            summary = f"States that {indicator} {key_phrase.strip()}"
            
        return {'summary': summary}
    else:
        # If no good key phrases found, create a brief general summary
        # Look for important terms in the text
        important_terms = ['right', 'duty', 'obligation', 'liability', 'penalty', 'punishment', 
                         'compensation', 'damages', 'relief', 'remedy', 'jurisdiction']
        
        found_terms = [term for term in important_terms if term in text_lower]
        
        if found_terms:
            # Use the first found term to create a brief summary
            term = found_terms[0]
            summary = f"Addresses {term} under {act_name_clean}"
        else:
            # Fallback to a very brief general summary
            summary = f"Contains legal provisions from {act_name_clean}"
            
        return {'summary': summary}

def parse_legal_response(text: str) -> Dict:
    sections = {
        'summary': "",
        'applicability': "",
        'key_points': [],
        'actions': []
    }
    
    current_section = None
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        if 'Summary:' in line:
            sections['summary'] = line.split('Summary:')[-1].strip()[:120]  # Limit length
        elif 'Applies when:' in line:
            sections['applicability'] = line.split('Applies when:')[-1].strip()[:60]
        elif 'Key points:' in line:
            current_section = 'key_points'
        elif 'Steps:' in line:
            current_section = 'actions'
        elif current_section and line.startswith('-') and len(sections[current_section]) < 2:
            sections[current_section].append(line[1:].strip()[:80])
    
    return sections

def is_follow_up(query: str, session: dict) -> bool:
    """Determine if the current query is a follow-up to previous conversation."""
    # If no previous context exists, it can't be a follow-up
    if not session.get('history') or len(session.get('history', [])) < 2:
        return False

    # Common follow-up phrases and patterns
    follow_up_indicators = [
        'follow up', 'previous', 'explain', 'more info', 'about that', 
        'tell me more', 'clarify', 'elaborate', 'continue', 'and then',
        'what about', 'how about', 'can you further', 'also', 'provide',
        'related', 'applicable', 'laws', 'sections', 'rights', 'help'
    ]

    # Very short queries are likely follow-ups (2-7 words)
    if 2 <= len(query.split()) <= 7:
        return True
        
    # Check for follow-up indicators
    if any(indicator in query.lower() for indicator in follow_up_indicators):
        return True
        
    # If the query contains words from the previous exchange, it's likely a follow-up
    prev_exchanges = session.get('history', [])
    if prev_exchanges:
        # Get the last few exchanges (up to 4 messages)
        recent_msgs = prev_exchanges[-4:] if len(prev_exchanges) >= 4 else prev_exchanges
        recent_text = " ".join([msg[1].lower() for msg in recent_msgs])
        
        # Get important words from the current query (excluding stopwords)
        query_words = set(query.lower().split())
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                    'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as', 'of'}
        query_keywords = query_words - stopwords
        
        # If several keywords from the query appear in recent conversation, it's a follow-up
        matches = sum(1 for word in query_keywords if word in recent_text)
        if matches >= 2 or (matches >= 1 and len(query_keywords) <= 4):
            return True
    
    return False

def analyze_query_context(query: str, conversation_history: str = "") -> Dict:
    """Analyze query context with minimal API usage"""
    try:
        # Extract key information from the query
        query_lower = query.lower()
        
        # Check for follow-up indicators
        follow_up_indicators = ['follow', 'previous', 'explain', 'more', 'about', 
                              'tell', 'clarify', 'elaborate', 'continue', 'and',
                              'what about', 'how about', 'also', 'provide', 'laws']
        
        is_follow_up = False
        # Short queries are likely follow-ups
        if len(query.split()) <= 7:
            is_follow_up = True
        # Check for follow-up terms
        elif any(indicator in query_lower for indicator in follow_up_indicators):
            is_follow_up = True
        
        # Check for legal-specific terms
        legal_terms = ['law', 'section', 'act', 'statute', 'legal', 'provision', 'right',
                      'court', 'judge', 'criminal', 'civil', 'lawsuit', 'sue', 'attorney',
                      'lawyer', 'case', 'precedent', 'legislation']
        
        # Check for informational or procedural query patterns
        is_informational = any(pattern in query_lower for pattern in [
            'how do i', 'how to', 'what is', 'what are', 'where can', 'when should',
            'process for', 'procedure', 'steps to', 'guide', 'explain how', 'file',
            'register', 'apply', 'submit', 'procedure for'
        ])
        
        # Determine if we have sufficient context
        context_sufficient = False
        
        if is_informational:
            context_sufficient = True
        elif is_follow_up:
            context_sufficient = True
        elif len(query.split()) > 25:
            context_sufficient = True
        elif "when:" in query_lower and "where:" in query_lower:
            context_sufficient = True
        
        # Create general follow-up questions
        follow_up_questions = [
            "When and where did this situation occur?",
            "What specific actions have you already taken?",
            "Are there any important dates or deadlines we should know about?",
            "Do you have any documentation or evidence related to this situation?",
            "Have you already discussed this with any legal professionals?"
        ]
        
        return {
            "is_follow_up": is_follow_up,
            "query_type": "legal",  # All queries are treated as legal
            "requires_sections": True,  # Always include sections for legal context
            "requires_cases": 'case' in query_lower or 'precedent' in query_lower,
            "context_sufficient": context_sufficient,
            "is_informational": is_informational,
            "follow_up_questions": follow_up_questions
        }
    except Exception as e:
        print(f"Local context analysis error: {e}")
        return {
            "is_follow_up": len(query.split()) <= 5,
            "query_type": "legal",
            "requires_sections": True,
            "requires_cases": False,
            "context_sufficient": False,
            "is_informational": False,
            "follow_up_questions": [
                "Could you provide more details about your situation?",
                "When and where did this occur?",
                "What steps have you already taken?",
                "Do you have any documentation related to this matter?",
                "Have you spoken with anyone about this issue already?"
            ]
        }


def format_response(base_answer: str, references: List[Dict], cases: List[Dict], query_type: str) -> str:
    # Extract steps from base answer and clean them
    steps = [line.strip() for line in base_answer.split('\n') if line.strip().startswith(('1', '2', '3'))]
    steps = [step.split('.', 1)[1].strip() if '.' in step else step for step in steps]
    
    response = "🚨 **Immediate Steps:**\n"
    for i, step in enumerate(steps[:3], 1):
        response += f"{i}. {step}\n"
    
    # Include legal provisions
    if references:
        response += "\n\n⚖️ **Relevant Legal Provisions:**\n\n"
        for ref in references[:3]:  # Changed from [:2] to [:3] to show all 3 sections
            act_name = ref['act'].replace('.pdf', '').strip()
            section_text = ref['summary'].strip()
            response += f"• {act_name} Section {ref['section_number']}:\n  {section_text}\n\n"
    
    # Updated key recommendations
    response += "\n\n📌 **Key Recommendations:**\n\n"
    recommendations = [
        "Maintain all documents, including FIR, insurance communication, and RTO receipts.",
        "Keep a record of all dates and updates.",
        "Preserve any potential evidence (CCTV, witness details).",
        "Consult a lawyer if the case becomes complex or delayed.",
        "Periodically check the vehicle theft records or blacklist status online."
    ]
    for rec in recommendations:
        response += f"• {rec}\n"
    
    # Include case law if available
    if cases:
        response += "\n\n🔗 **Related Case Law:**\n\n"
        for case in cases[:3]:
            case_title = case.get('title', 'Case title not available')
            case_url = case.get('url', '')
            if case_url:
                response += f"• [{case_title[:80]}]({case_url})"
                if case.get('snippet') and not case['snippet'].startswith(case_title):
                    snippet = case['snippet']
                    if len(snippet) > 100:
                        snippet = snippet[:100] + "..."
                    response += f"\n  {snippet}"
                response += "\n"
            else:
                response += f"• {case_title[:80]}\n"
    
    return response.strip()

def generate_case_law_response(query: str, conversation_history: str = "") -> str:
    """Generate a response specifically about case law related to the query"""
    # Extract context-aware keywords
    search_terms = extract_keywords_from_conversation(conversation_history, query)
    
    # Fetch relevant case law
    cases = fetch_kanoon_results(search_terms, conversation_history)
    
    if not cases:
        return "I couldn't find any directly relevant case law for your situation. For the most accurate legal advice regarding your specific case, I recommend consulting with a lawyer who specializes in this area of law."
    
    # Generate a response explaining the cases
    case_info = "\n\n".join([f"• {case['title']}\n  {case['snippet']}" for case in cases])
    
    prompt = f"""
    As a legal expert, explain how these case precedents might apply to this situation:
    
    Query: {query}
    
    Previous conversation:
    {conversation_history}
    
    Relevant case law:
    {case_info}
    
    Provide a clear explanation of how these cases might be relevant to the user's situation, any important principles they established, and how they might apply to the current case. Be specific and practical in your explanations.
    """
    
    try:
        response = generate_with_together(prompt)
        
        # Format the response with the cases
        formatted_response = "**Relevant Case Law:**\n\n" + response + "\n\n"
        
        # Add links to the cases
        formatted_response += "**Case References:**\n\n"
        for case in cases:
            formatted_response += f"• [{case['title']}]({case['url']})\n"
            
        return formatted_response
        
    except Exception as e:
        print(f"Case law response generation error: {e}")
        default_response = "**Relevant Case Law:**\n\n"
        default_response += "Based on the information provided, these cases may be relevant to your situation:\n\n"
        
        for case in cases:
            default_response += f"• [{case['title']}]({case['url']})\n"
            
        return default_response

def search_relevant_sections(query: str, query_type: str, conversation_history: str = "") -> List[Dict]:
    """Search for sections relevant to the query using semantic search with conversation context"""
    try:
        print(f"Searching for relevant sections for query: {query}")
        
        # Extract key terms from the query
        query_lower = query.lower()
        
        # Create a more focused search query
        search_terms = []
        
        # Add legal-specific terms if found in the query
        legal_terms = ['law', 'legal', 'right', 'duty', 'obligation', 'liability', 'contract', 
                      'agreement', 'court', 'judge', 'judgment', 'case', 'precedent', 'statute',
                      'act', 'section', 'clause', 'provision', 'regulation', 'rule']
        
        for term in legal_terms:
            if term in query_lower:
                search_terms.append(term)
        
        # Create a combined search query that includes conversation history
        enhanced_query = query
        if search_terms:
            enhanced_query = query + " " + " ".join(search_terms)
        
        # If we have conversation history, use it to enhance the search
        if conversation_history:
            # Extract the most recent exchange from conversation history
            history_lines = conversation_history.split('\n')
            recent_exchanges = []
            for line in history_lines:
                if line.startswith('User:') or line.startswith('Assistant:'):
                    recent_exchanges.append(line)
            
            # Get the last user query and assistant response
            last_exchange = []
            for line in reversed(recent_exchanges):
                if line.startswith('User:'):
                    last_exchange.insert(0, line.replace('User:', '').strip())
                elif line.startswith('Assistant:'):
                    last_exchange.insert(0, line.replace('Assistant:', '').strip())
                if len(last_exchange) >= 2:
                    break
            
            # Combine the last exchange with current query for better context
            if last_exchange:
                context_query = " ".join(last_exchange) + " " + enhanced_query
                print(f"Enhanced search query with context: {context_query}")
                
                # Use the context-enhanced query for embedding
                query_embedding = model.encode([context_query])
            else:
                query_embedding = model.encode([enhanced_query])
        else:
            query_embedding = model.encode([enhanced_query])
            
        print(f"Final search query: {enhanced_query}")
        
        # Use FAISS index for initial search
        D, I = index.search(query_embedding, 10)  # Get more candidates for filtering
        
        # Collect initial results
        candidates = []
        for i, idx in enumerate(I[0]):
            candidates.append({
                'act': section_data[idx]['act'],
                'section_number': section_data[idx]['section_number'],
                'full_text': section_data[idx]['full_text'],
                'score': D[0][i]
            })
        
        # Get the top results
        top_results = candidates[:5]
        
        # Generate summaries for each section
        results = []
        for section in top_results:
            try:
                # Check if we have a meaningful summary in the section data
                if 'summary' in section and section['summary']:
                    summary = section['summary']
                else:
                    # Generate a new summary
                    analysis = generate_legal_analysis(section['full_text'], section['act'], section['section_number'])
                    summary = analysis['summary']
                
                # Add the section with its summary
                results.append({
                    'act': section['act'],
                    'section_number': section['section_number'],
                    'full_text': section['full_text'],
                    'summary': summary
                })
            except Exception as e:
                print(f"Error processing section: {e}")
                # Add a default summary
                results.append({
                    'act': section['act'],
                    'section_number': section['section_number'],
                    'full_text': section['full_text'],
                    'summary': f"Contains relevant legal provisions from {section['act']}"
                })
        
        return results[:3]  # Return top 3 results
        
    except Exception as e:
        print(f"Error in section search: {e}")
        return []  # Return empty list on error

def analyze_legal_impact(query: str, conversation_history: str = "", context: str = "") -> str:
    """Analyze the potential impact of a legal issue using Together AI"""
    try:
        # Create a more structured and concise prompt for impact analysis
        impact_prompt = f"""
        As a legal expert, provide a CONCISE and STRUCTURED impact analysis for this legal situation:

        User's Situation: {query}
        Previous Context: {conversation_history if conversation_history else "No previous context"}
        Additional Context: {context if context else "No additional context provided"}

        Provide a BRIEF impact analysis in this EXACT format:

        **Immediate Impact:**
        • [2-3 bullet points of immediate consequences]

        **Financial Impact:**
        • [2-3 bullet points of financial implications]

        **Legal Impact:**
        • [2-3 bullet points of legal consequences]

        **Long-term Consequences:**
        • [2-3 bullet points of lasting effects]

        **Risk Level:** [Low/Medium/High]

        **Key Mitigation Steps:**
        • [3-4 actionable steps to minimize impact]

        Keep each section brief and focused. Use bullet points for clarity. Be specific but concise.
        """

        # Generate impact analysis using Together AI
        impact_analysis = generate_with_together(impact_prompt)
        
        if not impact_analysis or len(impact_analysis.strip()) < 50:
            # Fallback response if AI generation fails
            return """
**Immediate Impact:**
• Loss of property and disruption of daily activities
• Need to report to police and insurance immediately
• Emotional stress and time investment in recovery process

**Financial Impact:**
• Insurance deductible and potential premium increases
• Replacement costs for stolen items and vehicle
• Legal fees if case becomes complex

**Legal Impact:**
• Right to file police report and seek legal action
• Obligation to cooperate with investigation
• Potential involvement in legal proceedings

**Long-term Consequences:**
• Impact on insurance history and future premiums
• Potential effect on credit if vehicle was financed
• Changes to security measures and insurance coverage

**Risk Level:** Medium

**Key Mitigation Steps:**
• Report theft immediately to police and insurance
• Document all losses and keep detailed records
• Consider legal consultation for complex cases
• Review and update security measures
"""
        
        return impact_analysis.strip()
        
    except Exception as e:
        print(f"Error in impact analysis: {e}")
        return """
**Immediate Impact:**
• Immediate loss and disruption to daily routine
• Required reporting and documentation process
• Emotional and practical challenges

**Financial Impact:**
• Direct financial losses and replacement costs
• Insurance implications and potential premium changes
• Additional expenses for legal or recovery services

**Legal Impact:**
• Legal rights to pursue recovery and justice
• Obligations to cooperate with authorities
• Potential legal proceedings and requirements

**Long-term Consequences:**
• Lasting effects on insurance and financial standing
• Changes to security practices and risk management
• Potential impact on future legal matters

**Risk Level:** Medium to High

**Key Mitigation Steps:**
• Take immediate action to report and document
• Seek professional legal and insurance guidance
• Maintain comprehensive records of all actions
• Implement preventive measures for the future
"""

def detect_impact_query(query: str) -> bool:
    """Detect if the user is asking about the impact of their legal issue"""
    query_lower = query.lower()
    
    # Primary impact keywords that clearly indicate impact analysis requests
    primary_impact_keywords = [
        'impact', 'effect', 'consequence', 'result', 'outcome', 'implication',
        'what will happen', 'what happens if', 'how will this affect',
        'what are the consequences', 'what are the effects', 'what is the outcome',
        'how bad is this', 'how serious is this', 'what are the risks',
        'what could happen', 'what might happen', 'worst case scenario',
        'financial impact', 'legal impact', 'personal impact', 'professional impact',
        'long term effects', 'lasting consequences', 'future implications'
    ]
    
    # Check for primary impact keywords first
    for keyword in primary_impact_keywords:
        if keyword in query_lower:
            return True
    
    # Secondary impact indicators that need more context
    secondary_impact_indicators = [
        'penalty', 'punishment', 'fine', 'damage', 'loss', 'risk'
    ]
    
    # For secondary indicators, check if they're used in impact context
    for indicator in secondary_impact_indicators:
        if indicator in query_lower:
            # Check if it's used in an impact-related context
            impact_context_words = ['what', 'how', 'tell me about', 'explain', 'describe', 'analyze']
            for context_word in impact_context_words:
                if context_word in query_lower:
                    return True
    
    # Check for question patterns that suggest impact analysis
    impact_questions = [
        'what if', 'what about', 'how about', 'what about the',
        'tell me about', 'explain the', 'describe the', 'analyze the'
    ]
    
    for pattern in impact_questions:
        if pattern in query_lower:
            # Additional check to see if it's followed by impact-related terms
            words_after = query_lower.split(pattern)[-1].split()
            if any(word in ['impact', 'effect', 'consequence', 'result', 'outcome'] for word in words_after[:3]):
                return True
    
    return False

app = FastAPI()

router = APIRouter(prefix="/Nyayadoot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessQueryRequest(BaseModel):
    query: str
    session_id: str = "default"

class ResponseModel(BaseModel):
    answer: str
    references: List[Dict]
    cases: List[Dict]
    is_follow_up: bool
    session_id: str

@router.post("/process-query", response_model=ResponseModel)
async def process_legal_query(request: ProcessQueryRequest):
    print(f"Received session_id: {request.session_id}")
    try:
        # Generate a new session ID if it's "new_chat"
        if request.session_id == "new_chat":
            request.session_id = str(uuid.uuid4())
            print(f"Created new session: {request.session_id}")
        
        session = conv_state.get_session(request.session_id)
        query = sanitize_query(request.query)
        
        if len(query) < 3:
            raise HTTPException(status_code=400, detail="Please ask a more detailed question.")

        # Check if this is the first query in this session
        is_first_query = len(session.get('history', [])) == 0
        print(f"Is first query: {is_first_query}")

        # --- LEGAL QUERY CHECK ONLY FOR FIRST QUERY ---
        if is_first_query:
            if not is_legal_query_together(query):
                return ResponseModel(
                    answer="I am not trained for this thing.",
                    references=[],
                    cases=[],
                    is_follow_up=False,
                    session_id=request.session_id
                )
        # Get conversation history (limited to reduce token usage)
        conversation_history = conv_state.get_conversation_history(request.session_id, max_turns=2)
        print(f"\nConversation history length: {len(conversation_history)} characters")

        # Check if this is likely a follow-up question using our method
        is_followup_query = is_follow_up(query, session)
        print(f"Is follow-up (direct check): {is_followup_query}")

        # Analyze query context without API calls
        context_analysis = analyze_query_context(query, conversation_history)
        print(f"\nProcessing query: {query}")
        print(f"Is follow-up (from analysis): {context_analysis.get('is_follow_up', False)}")
        print(f"Context sufficient: {context_analysis['context_sufficient']}")
        
        # Use either method to determine if it's a follow-up
        is_followup = is_followup_query or context_analysis.get('is_follow_up', False)
        
        # If it's the first query, always ask for more details unless it's very specific
        # or if it's not a follow-up and doesn't have sufficient context
        should_ask_followup = (is_first_query and len(query.split()) < 20) or (not is_followup and not context_analysis["context_sufficient"])
        
        # But skip follow-up questions for informational queries
        if context_analysis.get("is_informational", False):
            should_ask_followup = False
            print("Informational query detected, skipping follow-up questions")
        
        if should_ask_followup and context_analysis.get("follow_up_questions"):
            questions = context_analysis["follow_up_questions"]
            
            # Make a more engaging prompt for follow-up questions
            formatted_response = f"""
**To better assist you with your question about "{query.strip()}", I need some additional information:**

{chr(10).join(f"• {q}" for q in questions[:5])}

These details will help me provide you with more accurate guidance tailored to your specific situation.
"""
            # Save this interaction in history
            conv_state.update(
                request.session_id, 
                query, 
                formatted_response.strip(),
                [], 
                []
            )
            
            return ResponseModel(
                    answer=formatted_response.strip(),
                    references=[],
                    cases=[],
                    is_follow_up=False,
                    session_id=request.session_id
            )

        # If we get here, we have sufficient context to generate a response
        try:
            # Initialize references and cases
            references = []
            cases = []

            # Check if user is asking about impact FIRST (this takes priority)
            is_impact_query = detect_impact_query(query)
            print(f"Is impact query: {is_impact_query}")

            # Only check for laws and cases if it's NOT an impact query
            if not is_impact_query:
                # Check if user explicitly asked for sections/laws
                explicitly_asked_for_laws = any(term in query.lower() for term in [
                    'law', 'section', 'act', 'statute', 'provision', 'legislation',
                    'legal provisions', 'laws', 'sections', 'regulations', 'rules',
                    'what laws', 'which laws', 'relevant laws', 'applicable laws'
                ])
                
                # Check if user explicitly asked for cases
                explicitly_asked_for_cases = any(term in query.lower() for term in [
                    'case', 'precedent', 'judgment', 'ruling', 'decision', 'court case',
                    'case law', 'legal cases', 'previous cases', 'similar cases',
                    'supreme court', 'high court', 'court ruling',
                    'what cases', 'which cases', 'relevant cases', 'any cases'
                ])

                # Get relevant sections if requested
                if explicitly_asked_for_laws:
                    print("User explicitly asked for laws/sections")
                    try:
                        # Use find_relevant_sections instead of search_relevant_sections
                        sections = find_relevant_sections(query, conversation_history if is_followup else "")
                        if sections:
                            for section in sections[:3]:
                                try:
                                    summary = section.get('summary', "Relevant legal provision")
                                    references.append({
                                        'act': section['act'],
                                        'section_number': section['section_number'],
                                        'summary': summary,
                                        'full_text': section['full_text'][:250] + '...'
                                    })
                                except Exception as section_error:
                                    print(f"Error processing section: {section_error}")
                    except Exception as sections_error:
                        print(f"Error finding sections: {sections_error}")

                # Get case law if requested
                if explicitly_asked_for_cases:
                    print("User explicitly asked for cases/precedents")
                    try:
                        # Use generate_case_law_response instead of direct fetch_kanoon_results
                        case_law_response = generate_case_law_response(query, conversation_history if is_followup else "")
                        # Extract cases from the response
                        cases = fetch_cases_from_api_suggestions(case_law_response)
                    except Exception as cases_error:
                        print(f"Error fetching cases: {cases_error}")
                        cases = []

            # Handle impact analysis if requested
            impact_analysis = ""
            if is_impact_query:
                print("User is asking about impact analysis")
                try:
                    # Get relevant context from conversation history and any found sections
                    context_parts = []
                    if conversation_history:
                        context_parts.append(f"Previous conversation: {conversation_history}")
                    
                    if references:
                        ref_context = "Relevant legal provisions found: " + "; ".join([
                            f"{ref['act']} Section {ref['section_number']}" for ref in references[:2]
                        ])
                        context_parts.append(ref_context)
                    
                    context = " ".join(context_parts)
                    
                    # Generate impact analysis
                    impact_analysis = analyze_legal_impact(query, conversation_history, context)
                    print("Impact analysis generated successfully")
                    
                    # For impact queries, return only the impact analysis without other components
                    formatted_answer = f"""
🎯 **Impact Analysis**

{impact_analysis}

**Next Steps:**
1. Consult with a qualified lawyer for personalized advice
2. Document all relevant information and evidence
3. Follow legal guidance promptly to minimize negative impact
4. Keep records of all communications and actions taken

*This analysis is for informational purposes only and should not replace professional legal counsel.*
"""
                    
                    # Update conversation history
                    conv_state.update(request.session_id, query, formatted_answer.strip(), [], [])
                    
                    return ResponseModel(
                        answer=formatted_answer.strip(),
                        references=[],
                        cases=[],
                        is_follow_up=is_followup,
                        session_id=request.session_id
                    )
                    
                except Exception as impact_error:
                    print(f"Error generating impact analysis: {impact_error}")
                    impact_analysis = "I'm unable to provide a detailed impact analysis at this moment. Please consult with a qualified lawyer for a comprehensive assessment of your situation."
                    
                    # Return fallback impact response
                    fallback_response = f"""
🎯 **Impact Analysis**

{impact_analysis}

**Why professional consultation is important:**
- Legal situations can have complex, long-term consequences
- Each case has unique circumstances that affect outcomes
- Professional advice can help minimize negative impact
- Lawyers can provide specific strategies for your situation

For immediate guidance, please consult with a legal professional who can provide personalized advice based on your specific circumstances.
"""
                    
                    conv_state.update(request.session_id, query, fallback_response.strip(), [], [])
                    
                    return ResponseModel(
                        answer=fallback_response.strip(),
                        references=[],
                        cases=[],
                        is_follow_up=is_followup,
                        session_id=request.session_id
                    )

            # For non-impact queries, generate base answer and format response
            base_answer = generate_direct_answer(query, conversation_history=conversation_history, is_followup=is_followup)
            
            # Format the response using the format_response helper
            formatted_answer = format_response(base_answer, references, cases, "legal")

            # Add impact analysis to the response if available (for non-impact queries that might benefit from impact info)
            if impact_analysis:
                formatted_answer += f"\n\n🎯 **Impact Analysis:**\n\n{impact_analysis}"

            # Parse the response to ensure proper structure
            parsed_response = parse_legal_response(formatted_answer)
            
            # Update conversation history
            conv_state.update(request.session_id, query, formatted_answer.strip(), references, cases)
        
            return ResponseModel(
                answer=formatted_answer.strip(),
                references=references,
                cases=cases,
                is_follow_up=is_followup,
                session_id=request.session_id
            )
        except Exception as inner_e:
            print(f"Error processing query: {inner_e}")
            # Fallback response with improved formatting
            fallback_answer = f"""
Based on your query about "{query}", here are some general recommendations:

1. Document all relevant information and evidence related to your situation
2. Consider consulting with a qualified lawyer who specializes in this area
3. Keep records of all related communications and events

For specific legal advice tailored to your situation, please consult with a legal professional.
"""
            conv_state.update(request.session_id, query, fallback_answer, [], [])
            
            return ResponseModel(
                answer=fallback_answer,
                references=[],
                cases=[],
                is_follow_up=is_followup_query,
                session_id=request.session_id
            )
    except Exception as outer_e:
        print(f"Outer exception: {outer_e}")
        raise HTTPException(status_code=500, detail=str(outer_e))

@router.post("/reset-session")
async def reset_session(session_id: str = Query(...)):
    if session_id in conv_state.sessions:
        del conv_state.sessions[session_id]
    return {"status": "reset"}

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)