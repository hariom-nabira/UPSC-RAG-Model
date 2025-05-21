import React, { useState, useEffect } from 'react';
import './index.css';
import MessageDisplayArea from './components/MessageDisplayArea';
import MessageInputArea from './components/MessageInputArea';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1/chat';
const SESSION_STORAGE_KEY = 'chatSessionId';
const CHAT_SESSIONS_KEY = 'chatSessionsList'; // Key for storing the list of sessions

function App() {
  const [messages, setMessages] = useState([
    { id: 'initial-ai-greeting', text: 'Hello! How can I help you today with your UPSC preparation?', sender: 'ai' },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentSessionId, setCurrentSessionId] = useState(() => {
    return localStorage.getItem(SESSION_STORAGE_KEY);
  });
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [chatSessions, setChatSessions] = useState([]);

  // Load chat sessions from localStorage on mount
  useEffect(() => {
    const storedSessions = localStorage.getItem(CHAT_SESSIONS_KEY);
    if (storedSessions) {
      try {
        setChatSessions(JSON.parse(storedSessions));
      } catch (e) {
        console.error("Failed to parse chat sessions from localStorage:", e);
        localStorage.removeItem(CHAT_SESSIONS_KEY); // Clear corrupted data
      }
    }
    // Check system preference for dark mode
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    setIsDarkMode(prefersDark);
  }, []);

  // Effect to save current session ID to localStorage whenever it changes
  useEffect(() => {
    if (currentSessionId) {
      localStorage.setItem(SESSION_STORAGE_KEY, currentSessionId);
    } else {
      localStorage.removeItem(SESSION_STORAGE_KEY);
    }
  }, [currentSessionId]);

  // Effect to apply dark mode class to root element
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  const formatChatHistoryForAPI = (history) => {
    return history.map(msg => ({ role: msg.sender, content: msg.text }));
  };

  const upsertChatSession = (sessionIdToUpdate, titleText) => {
    setChatSessions(prevSessions => {
      const existingSessionIndex = prevSessions.findIndex(s => s.id === sessionIdToUpdate);
      let newSessions;
      const sessionTitle = titleText ? titleText.substring(0, 40) + (titleText.length > 40 ? '...' : '') : 'New Chat';
      
      if (existingSessionIndex > -1) {
        newSessions = [...prevSessions];
        newSessions[existingSessionIndex] = {
          ...newSessions[existingSessionIndex],
          title: newSessions[existingSessionIndex].title === 'New Chat' && titleText ? sessionTitle : newSessions[existingSessionIndex].title,
          timestamp: Date.now(),
        };
      } else {
        newSessions = [{ id: sessionIdToUpdate, title: sessionTitle, timestamp: Date.now() }, ...prevSessions];
      }
      newSessions.sort((a, b) => b.timestamp - a.timestamp);
      localStorage.setItem(CHAT_SESSIONS_KEY, JSON.stringify(newSessions));
      return newSessions;
    });
  };

  const handleSendMessage = async (newMessageText) => {
    const userMessage = { id: Date.now().toString(), text: newMessageText, sender: 'user' };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setIsLoading(true);
    setError(null);

    const wasNewSession = !currentSessionId;
    const currentAiMessageId = Date.now().toString() + '-ai-stream';
    const aiMessagePlaceholder = { id: currentAiMessageId, text: '', sender: 'ai', sources: [], isLoading: true };
    setMessages(prevMessages => [...prevMessages, aiMessagePlaceholder]);

    const historyForAPI = !currentSessionId ? formatChatHistoryForAPI(messages) : [];

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
        body: JSON.stringify({ message: newMessageText, session_id: currentSessionId, chat_history: historyForAPI }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: 'Streaming API connection error' }));
        throw new Error(errData.detail || `API Error: ${response.status} ${response.statusText}`);
      }

      let receivedSessionIdThisStream = null;
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let stillStreaming = true;

      while (stillStreaming) {
        const { value, done } = await reader.read();
        if (done) { stillStreaming = false; break; }
        const chunk = decoder.decode(value);
        const eventLines = chunk.split('\n\n').filter(line => line.trim() !== '');

        for (const line of eventLines) {
          if (line.startsWith('data: ')) {
            const jsonData = line.substring(6);
            try {
              const eventData = JSON.parse(jsonData);
              if (eventData.type === 'session_id') {
                receivedSessionIdThisStream = eventData.id;
                setCurrentSessionId(eventData.id);
                if (wasNewSession || !chatSessions.find(s => s.id === eventData.id)) {
                  upsertChatSession(eventData.id, newMessageText);
                } else {
                  upsertChatSession(eventData.id);
                }
              } else if (eventData.type === 'token') {
                setMessages(prev => prev.map(m => m.id === currentAiMessageId ? { ...m, text: m.text + eventData.content, isLoading: true } : m));
              } else if (eventData.type === 'sources') {
                setMessages(prev => prev.map(m => m.id === currentAiMessageId ? { ...m, sources: eventData.content } : m));
              } else if (eventData.type === 'end') {
                setMessages(prev => prev.map(m => m.id === currentAiMessageId ? { ...m, isLoading: false } : m));
                stillStreaming = false; break;
              } else if (eventData.type === 'error') {
                setError(eventData.content);
                setMessages(prev => prev.map(m => m.id === currentAiMessageId ? { ...m, text: (m.text || "") + `\nStream Error: ${eventData.content}`, isLoading: false, isError: true } : m));
                stillStreaming = false; break;
              }
            } catch (e) {
              console.error("Failed to parse stream event data:", jsonData, e);
              setError("Failed to process some stream data.");
              setMessages(prev => prev.map(m => m.id === currentAiMessageId ? { ...m, text: (m.text || "") + `\nError parsing stream data.`, isLoading: false, isError: true } : m));
              stillStreaming = false; break;
            }
          }
        }
      }
      if (currentSessionId && !receivedSessionIdThisStream) {
         upsertChatSession(currentSessionId);
      }
    } catch (err) {
      console.error("Failed to send message or process stream:", err);
      setError(err.message);
      setMessages(prevMessages => {
        const aiMessageIndex = prevMessages.findIndex(m => m.id === currentAiMessageId);
        if (aiMessageIndex !== -1) {
          const currentMessage = prevMessages[aiMessageIndex];
          if (currentMessage.text === '') { return prevMessages.filter(m => m.id !== currentAiMessageId); }
          return prevMessages.map((msg, idx) => idx === aiMessageIndex ? { ...msg, text: msg.text + `\nRequest Error: ${err.message}`, isLoading: false, isError: true } : msg);
        }
        return [...prevMessages, { id: currentAiMessageId + '-fetch-error', text: `Error: ${err.message}`, sender: 'ai', isError: true, isLoading: false }];
      });
    } finally {
      setIsLoading(false);
      setMessages(prevMessages => prevMessages.map(msg => msg.id === currentAiMessageId ? { ...msg, isLoading: false } : msg));
    }
  };

  const clearSession = () => {
    const oldSessionId = currentSessionId;
    if (oldSessionId && chatSessions.find(s => s.id === oldSessionId)) {
        upsertChatSession(oldSessionId);
    }
    setCurrentSessionId(null);
    setMessages([
        { id: 'initial-ai-greeting-cleared', text: 'New session started. How can I help?', sender: 'ai' },
    ]);
    setError(null);
  };
  
  const handleSelectSession = (sessionIdToSelect) => {
    if (currentSessionId === sessionIdToSelect) return;
    if (currentSessionId && chatSessions.find(s => s.id === currentSessionId)){
        upsertChatSession(currentSessionId);
    }
    setCurrentSessionId(sessionIdToSelect);
    setMessages([
      { id: 'switched-session-greeting', text: `Switched to a previous session. Send a message to continue or review past interactions if they were loaded.`, sender: 'ai' }
    ]);
    setError(null);
  };

  const toggleDarkMode = () => setIsDarkMode(!isDarkMode);

  const formatDate = (timestamp) => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    if (date.toDateString() === today.toDateString()) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    if (date.toDateString() === yesterday.toDateString()) {
      return 'Yesterday';
    }
    return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
  };

  return (
    <div className={`flex h-screen antialiased text-gray-800 dark:text-gray-200 bg-white dark:bg-gray-900`}>
      {/* Sidebar */}
      <div className="flex flex-col w-64 bg-gray-50 dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 shrink-0">
        <div className="flex items-center justify-between h-16 border-b border-gray-200 dark:border-gray-700 p-4">
          <h1 className="text-lg font-semibold text-gray-700 dark:text-gray-200">Chat Sessions</h1>
          <button 
            onClick={clearSession} 
            title="Start New Chat"
            className="p-1.5 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400"
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v6m3-3H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </button>
        </div>
        <div className="flex-grow p-2 overflow-y-auto space-y-1 scrollbar-thin">
          {chatSessions.length === 0 && (
            <p className="text-xs text-gray-400 dark:text-gray-500 px-2 py-4 text-center">No past chats yet.</p>
          )}
          {chatSessions.map(session => (
            <button
              key={session.id}
              onClick={() => handleSelectSession(session.id)}
              title={session.title}
              className={`w-full text-left px-3 py-2.5 rounded-lg text-sm 
                          hover:bg-gray-100 dark:hover:bg-gray-700 
                          focus:outline-none focus:ring-1 focus:ring-indigo-500
                          transition-colors duration-100
                          ${currentSessionId === session.id ? 'bg-indigo-100 dark:bg-indigo-600 text-indigo-700 dark:text-indigo-50 font-semibold' 
                                                          : 'text-gray-600 dark:text-gray-300'}`}
            >
              <div className="truncate font-medium">{session.title}</div>
              <div className={`text-xs truncate ${currentSessionId === session.id ? 'text-indigo-500 dark:text-indigo-200' : 'text-gray-400 dark:text-gray-500'}`}>
                {formatDate(session.timestamp)}
              </div>
            </button>
          ))}
        </div>
        <div className="p-3 border-t border-gray-200 dark:border-gray-700">
            <button 
                onClick={toggleDarkMode}
                title={isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
                className="w-full flex items-center justify-center p-2 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-300 transition-colors duration-150"
            >
                {isDarkMode ? (
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    {/* Sun icon */}
                    <path fillRule="evenodd" d="M10 2a.75.75 0 01.75.75v1.5a.75.75 0 01-1.5 0v-1.5A.75.75 0 0110 2zM10 15a.75.75 0 01.75.75v1.5a.75.75 0 01-1.5 0v-1.5A.75.75 0 0110 15zM10 6a4 4 0 100 8 4 4 0 000-8zM1.5 9.25a.75.75 0 01.75-.75h1.5a.75.75 0 010 1.5h-1.5a.75.75 0 01-.75-.75zm14.5 0a.75.75 0 01.75-.75h1.5a.75.75 0 010 1.5h-1.5a.75.75 0 01-.75-.75zM4.22 4.22a.75.75 0 011.06 0l1.06 1.06a.75.75 0 01-1.06 1.06l-1.06-1.06a.75.75 0 010-1.06zm10.498 10.498a.75.75 0 011.06 0l1.06 1.06a.75.75 0 01-1.06 1.06l-1.06-1.06a.75.75 0 010-1.06zM4.22 15.77a.75.75 0 010-1.06l1.06-1.06a.75.75 0 111.06 1.06l-1.06 1.06a.75.75 0 01-1.06 0zm10.498-10.498a.75.75 0 010-1.06l1.06-1.06a.75.75 0 111.06 1.06l-1.06 1.06a.75.75 0 01-1.06 0z" clipRule="evenodd" />
                  </svg> 
                ) : (
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    {/* Moon icon */}
                    <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
                  </svg>
                )}
            </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex flex-col flex-grow min-w-0"> 
        <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 text-center shadow-sm h-16 flex items-center justify-center shrink-0">
          <h1 className="text-lg font-semibold text-gray-700 dark:text-gray-200">UPSC Dynamic Knowledge Engine</h1>
        </header>
        
        {error && (
          <div className="bg-red-100 border-l-4 border-red-500 text-red-700 dark:text-red-300 p-3 m-2 rounded-md text-sm shrink-0" role="alert">
            <p className="font-bold dark:text-red-200">Error</p>
            <p className="dark:text-red-200">{error}</p>
          </div>
        )}

        <MessageDisplayArea messages={messages} isLoading={isLoading && messages.every(m => !m.isLoading)} />
        <MessageInputArea onSendMessage={handleSendMessage} isLoading={isLoading} />
      </div>
    </div>
  );
}

export default App; 