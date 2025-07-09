import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Chat from './components/Chat';
import History from './components/History';
import LandingPage from './components/LandingPage';
import Captcha from './components/Captcha';

function App() {
  return (
        <Routes>
          <Route path="/" element={<LandingPage />} />
        <Route path="/captcha" element={<Captcha />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/history" element={<History />} />
        </Routes>
  );
}

export default App;