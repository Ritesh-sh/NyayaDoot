import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Chat from './components/Chat';
import History from './components/History';
import LandingPage from './components/LandingPage';
import Captcha from './components/Captcha';

function App() {
  return (
    <Routes>
      <Route path="/nyayadoot/" element={<LandingPage />} />
      <Route path="/nyayadoot/captcha" element={<Captcha />} />
      <Route path="/nyayadoot/chat" element={<Chat />} />
      <Route path="/nyayadoot/history" element={<History />} />
      <Route path="/" element={<Navigate to="/nyayadoot/" replace />} />
    </Routes>
  );
}

export default App;