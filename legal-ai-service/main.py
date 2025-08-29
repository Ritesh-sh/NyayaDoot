from fastapi import FastAPI, HTTPException, Query, APIRouter
from pydantic import BaseModel
from typing import List, Dict, Optional
import re
import html
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import uuid
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

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
    """Use Gemini to suggest relevant Act/Section pairs directly (no local mapping)."""
    try:
        context = f"Previous: {conversation_history}\n" if conversation_history else ""
        prompt = (
            "You are a legal assistant for Indian law. Given the user's question, "
            "suggest up to 3 highly relevant statute sections as ActName|SectionNumber|Why.\n"
            "Only output lines in this exact pipe-delimited format, no extra text.\n"
            f"{context}Question: {query}"
        )
        raw = generate_with_gemini(prompt)
        lines = [l.strip() for l in raw.split('\n') if '|' in l]

        suggestions = []
        for line in lines:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 2:
                act_s, sec_s = parts[0], parts[1]
                why = parts[2] if len(parts) >= 3 else "Relevant legal provision"
                suggestions.append((act_s, sec_s, why))

        results: List[Dict] = []
        for act_s, sec_s, why in suggestions:
            results.append({
                'act': act_s,
                'section_number': sec_s,
                'summary': why
            })
            if len(results) >= 3:
                break
                
        return results
    except Exception as e:
        print(f"Gemini section retrieval error: {e}")
        return []


def fetch_kanoon_results(query: str, conversation_history: str = "") -> List[Dict]:
    """Fetch case law results from Indian Kanoon using a Gemini-generated search phrase and filter for relevance. Retry on timeout."""
    import requests
    from bs4 import BeautifulSoup
    import time
    # Improve Gemini prompt for more relevant case law search
    context = f"Query: {query}\nHistory: {conversation_history}" if conversation_history else query
    prompt = (
        "You are a legal assistant for Indian law. Given the user's question and context, generate a search phrase that will find the most relevant and diverse Indian case law on Indian Kanoon. Focus on legal principles, parties, and jurisdiction. Only output the search phrase, no extra text.\n\n"
        f"{context}"
    )
    try:
        search_phrase = generate_with_gemini(prompt)
        print(f"Gemini search phrase: {search_phrase}")
        # Fallback to original query if Gemini output is too generic or short
        if not search_phrase or len(search_phrase.strip()) < 5:
            search_phrase = query
    except Exception as e:
        print(f"Gemini search phrase error: {e}")
        search_phrase = query
    url = f"https://indiankanoon.org/search/?formInput={requests.utils.quote(search_phrase)}"
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, timeout=20)
            soup = BeautifulSoup(resp.text, "html.parser")
            case_elements = soup.select("div.result_title > a")
            case_results = []
            seen_titles = set()
            seen_urls = set()
            for element in case_elements:
                title = element.get_text(strip=True)
                case_url = element.get("href")
                if case_url and not case_url.startswith("http"):
                    case_url = f"https://indiankanoon.org{case_url}"
                snippet = ""
                snippet_element = element.find_parent("div", class_="result_title").find_next_sibling("div", class_="snippet")
                if snippet_element:
                    snippet = snippet_element.get_text(strip=True)
                else:
                    snippet = title
                # Filter out duplicate cases by title and URL
                title_key = title.lower().replace("...", "").strip()
                if title_key in seen_titles or (case_url and case_url in seen_urls):
                    continue
                case_results.append({
                    "title": title[:80],
                    "url": case_url,
                    "snippet": snippet[:250] if snippet else ""
                })
                seen_titles.add(title_key)
                if case_url:
                    seen_urls.add(case_url)
                if len(case_results) == 3:
                    break
            if case_results:
                return case_results
            else:
                if attempt < max_retries:
                    time.sleep(2)
                else:
                    return [{
                        "title": "Indian Kanoon is currently unavailable",
                        "url": "https://indiankanoon.org/",
                        "snippet": "Sorry, we could not retrieve case law results at this time. Please try again later."
                    }]
        except requests.exceptions.Timeout:
            print(f"Kanoon timeout on attempt {attempt+1}")
            if attempt < max_retries:
                time.sleep(2)
            else:
                return [{
                    "title": "Indian Kanoon is currently unavailable",
                    "url": "https://indiankanoon.org/",
                    "snippet": "Sorry, we could not retrieve case law results due to a timeout. Please try again later."
                }]
        except Exception as e:
            print(f"Kanoon error: {e}")
            return [{
                "title": "Indian Kanoon is currently unavailable",
                "url": "https://indiankanoon.org/",
                "snippet": "Sorry, we could not retrieve case law results due to a technical error. Please try again later."
            }]

def fetch_specific_case_from_kanoon(case_name: str) -> Dict:
    """Search for a specific case name on Indian Kanoon and return the most relevant result"""
    print(f"Searching for specific case: {case_name}")
    import requests
    from bs4 import BeautifulSoup
    print(f"Searching for specific case: {case_name}")
    try:
        search_query = f'"{case_name}"'
        url = f"https://indiankanoon.org/search/?formInput={requests.utils.quote(search_query)}"
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        case_elements = soup.select("div.result_title > a")
        if case_elements:
            element = case_elements[0]
            title = element.get_text(strip=True)
            url = element.get("href")
            if url and not url.startswith("http"):
                url = f"https://indiankanoon.org{url}"
            snippet = ""
            snippet_element = element.find_parent("div", class_="result_title").find_next_sibling("div", class_="snippet")
            if snippet_element:
                snippet = snippet_element.get_text(strip=True)
            else:
                snippet = title
            return {
                "title": title[:80],
                "url": url,
                "snippet": snippet[:250] if snippet else "",
                "case_name": case_name
            }
        else:
            # Try without quotes if no results found
            url = f"https://indiankanoon.org/search/?formInput={requests.utils.quote(case_name)}"
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            case_elements = soup.select("div.result_title > a")
            if case_elements:
                element = case_elements[0]
                title = element.get_text(strip=True)
                url = element.get("href")
                if url and not url.startswith("http"):
                    url = f"https://indiankanoon.org{url}"
                snippet = ""
                snippet_element = element.find_parent("div", class_="result_title").find_next_sibling("div", class_="snippet")
                if snippet_element:
                    snippet = snippet_element.get_text(strip=True)
                else:
                    snippet = title
                return {
                    "title": title[:80],
                    "url": url,
                    "snippet": snippet[:250] if snippet else "",
                    "case_name": case_name
                }
        return {
            "title": f"Case: {case_name}",
            "url": f"https://indiankanoon.org/search/?formInput={case_name.replace(' ', '+')}",
            "snippet": f"This case was mentioned in the legal analysis but couldn't be found directly on Indian Kanoon.",
            "case_name": case_name
        }
    except Exception as e:
        print(f"Error searching for specific case: {e}")
        return {
            "title": f"Case: {case_name}",
            "url": f"https://indiankanoon.org/search/?formInput={case_name.replace(' ', '+')}",
            "snippet": "Could not retrieve case details due to technical issues.",
            "case_name": case_name
        }

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
    """Instead of searching for specific case names, use extracted keywords to fetch top 3 cases from Indian Kanoon."""
    # Extract keywords from the API response
    keywords = extract_keywords_from_conversation("", api_response)
    # Fetch top 3 cases using keyword search
    results = fetch_kanoon_results(keywords)
    return results[:3]  # Return up to 3 results

def generate_with_gemini(prompt: str) -> str:
    """Generate text using Gemini API with fallback."""
    try:
        if len(prompt) > 1000:
            prompt_lines = prompt.split('\n')
            if len(prompt_lines) > 20:
                prompt = '\n'.join(prompt_lines[:10] + ['\n...[content trimmed]...\n'] + prompt_lines[-10:])
            elif len(prompt) > 6000:
                prompt = prompt[:1500] + "\n...[content trimmed]...\n" + prompt[-1500:]
        print(f"Prompt length: {len(prompt)} characters")
        return gemini_generate(prompt)
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm having trouble connecting to my knowledge source. For legal matters, it's always best to consult with a qualified attorney who can provide personalized advice."

# Replace Together AI legal classifier with Gemini

def is_legal_query_gemini(query: str) -> bool:
    """
    Uses Gemini to determine if the query is legal in nature.
    Returns True if legal, False otherwise.
    """
    prompt = f"You are a classifier. Decide if the following user query is a legal question (about laws, rights, legal procedures, court cases, contracts, etc). Reply with only 'LEGAL' or 'NOT LEGAL'.\n\nQuery: \"{query}\"\n"
    try:
        result = gemini_generate(prompt, max_tokens=5, temperature=0.0).strip().upper()
        return result == "LEGAL"
    except Exception as e:
        print(f"Error in legal query classification: {e}")
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
    answer = generate_with_gemini(prompt)
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
        elif len(query) > 25:
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
    
    # Include case law if available, but filter out fallback/technical cases and duplicates
    if cases:
        filtered_cases = []
        seen_urls = set()
        for case in cases:
            snippet = case.get('snippet', '').lower()
            url = case.get('url', '')
            if ("couldn't be found directly on indian kanoon" in snippet or
                "could not retrieve case details due to technical issues" in snippet or
                "this case was mentioned in the legal analysis" in snippet):
                continue  # skip fallback/technical cases
            if url and url in seen_urls:
                continue  # skip duplicate cases by URL
            filtered_cases.append(case)
            if url:
                seen_urls.add(url)
        if filtered_cases:
            response += "\n\n🔗 **Related Case Law:**\n\n"
            for case in filtered_cases[:3]:
                case_title = case.get('title', 'Case title not available')
                case_url = case.get('url', '')
                if case_url:
                    # Clean up the case title to show only the case name (remove extra parties, dates, etc.)
                    # Try to extract a pattern like 'X vs Y (Year)' or 'X vs Y'
                    import re
                    clean_title = case_title
                    match = re.search(r'([A-Za-z][A-Za-z\s.&]+\s+vs\.?\s+[A-Za-z][A-Za-z\s.&]+)(?:\s*\(\d{4}\))?', case_title)
                    if match:
                        clean_title = match.group(0).strip()
                    response += f"• [{clean_title}]({case_url})"
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
        response = generate_with_gemini(prompt)
        
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
    """Wrapper to use Gemini-based retrieval for sections."""
    try:
        return find_relevant_sections(query, conversation_history)
    except Exception as e:
        print(f"Error in section search: {e}")
        return []  # Return empty list on error

def analyze_legal_impact(query: str, conversation_history: str = "", context: str = "") -> str:
    """Analyze the potential impact in a compact, token-efficient format."""
    try:
        # Token-lean, compact prompt (strict headings, minimal bullets)
        impact_prompt = f"""
        You are a senior Indian legal expert. Provide a COMPACT impact note.

        Situation: {query}
        Context: {conversation_history if conversation_history else "-"}
        Extra: {context if context else "-"}

        Output exactly these sections with at most the specified bullets:
        Summary: [max 2 line]
        Risk: [Low/Medium/High] - [<=8 words why]
        Immediate: - [max 2 bullets]
        Financial: - [max 2 bullets]
        Legal: - [max 2 bullets]
        Next: 
        1) [step] 
        2) [step] 
        3) [step]


        Do NOT add extra blank lines between sections.  
        Only bullets ( - ) for items, not for section headers.  
        Each point must be concise, clear, and on a new line.  
        Do not add any other text or formatting.  
        Do not add any other text or formatting.  
        """

        # Generate impact analysis using Gemini (short output)
        impact_analysis = gemini_generate(impact_prompt, max_tokens=220, temperature=0.2)
        
        if not impact_analysis or len(impact_analysis.strip()) < 50:
            # Fallback response if AI generation fails
            return """
Summary: Immediate reporting and documentation likely required.
Risk: Medium - deadlines/documentation
Immediate:
- Report via official channel
- Preserve evidence/documents
Financial:
- Possible fees/penalties
- Advisory/processing costs
Legal:
- Compliance and notices
- Potential proceedings
Next: 1) File report 2) Organize proofs 3) Track refs & deadlines
"""
        
        return impact_analysis.strip()
        
    except Exception as e:
        print(f"Error in impact analysis: {e}")
        return """
Summary: Prompt report + solid documentation advised.
Risk: Medium-High - time/accuracy critical
Immediate:
- File official notice/report
- Secure evidence & references
Financial:
- Direct/penalty costs possible
- Advisory fees likely if escalated
Legal:
- Compliance and responses expected
- Risk from missed timelines
Next: 1) File 2) Organize proofs 3) Track refs/deadlines
"""

def detect_impact_query(query: str) -> bool:
    """Detect if the user is asking about the impact of their legal issue"""
    query_lower = query.lower()
    primary_impact_keywords = [
        'impact', 'effect', 'consequence', 'result', 'outcome', 'implication',
        'what will happen', 'what happens if', 'how will this affect',
        'what are the consequences', 'what are the effects', 'what is the outcome',
        'how bad is this', 'how serious is this', 'what are the risks',
        'what could happen', 'what might happen', 'worst case scenario',
        'financial impact', 'legal impact', 'personal impact', 'professional impact',
        'long term effects', 'lasting consequences', 'future implications'
    ]
    for keyword in primary_impact_keywords:
        if keyword in query_lower:
            return True
    secondary_impact_indicators = [
        'penalty', 'punishment', 'fine', 'damage', 'loss', 'risk'
    ]
    for indicator in secondary_impact_indicators:
        if indicator in query_lower:
            impact_context_words = ['what', 'how', 'tell me about', 'explain', 'describe', 'analyze']
            for context_word in impact_context_words:
                if context_word in query_lower:
                    return True
    impact_questions = [
        'what if', 'what about', 'how about', 'what about the',
        'tell me about', 'explain the', 'describe the', 'analyze the'
    ]
    for pattern in impact_questions:
        if pattern in query_lower:
            words_after = query_lower.split(pattern)[-1].split()
            if any(word in ['impact', 'effect', 'consequence', 'result', 'outcome'] for word in words_after[:3]):
                return True
    return False

# Gemini API helper function

def gemini_generate(prompt: str, max_tokens: int = 256, temperature: float = 0.5) -> str:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        response = model.generate_content(prompt, generation_config={
            "max_output_tokens": max_tokens,
            "temperature": temperature
        })
        return response.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return "I'm having trouble connecting to my knowledge source. For legal matters, it's always best to consult with a qualified attorney who can provide personalized advice."

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
            if not is_legal_query_gemini(query):
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
                                        'summary': summary
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