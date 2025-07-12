import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Chat from './components/Chat';
import History from './components/History';
import LandingPage from './components/LandingPage';
import Captcha from './components/Captcha';

function App() {
  return (
    <Routes>
      <Route path="/Nyayadoot/" element={<LandingPage />} />
      <Route path="/Nyayadoot/captcha" element={<Captcha />} />
      <Route path="/Nyayadoot/chat" element={<Chat />} />
      <Route path="/Nyayadoot/history" element={<History />} />
      <Route path="/" element={<Navigate to="/Nyayadoot/" replace />} />
    </Routes>
  );
}

export default App;