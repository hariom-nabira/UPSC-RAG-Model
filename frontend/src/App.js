import React, { useState, useEffect } from 'react';
import './index.css';
import MessageDisplayArea from './components/MessageDisplayArea';
import MessageInputArea from './components/MessageInputArea';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1/chat';
const SESSION_STORAGE_KEY = 'chatSessionId';

function App() {
  const [messages, setMessages] = useState([
    { id: 'initial-ai-greeting', text: 'Hello! How can I help you today with your UPSC preparation?', sender: 'ai' },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [sessionId, setSessionId] = useState(() => {
    // Try to get session ID from localStorage on initial load
    return localStorage.getItem(SESSION_STORAGE_KEY);
  });

  // Effect to save session ID to localStorage whenever it changes
  useEffect(() => {
    if (sessionId) {
      localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
    } else {
      localStorage.removeItem(SESSION_STORAGE_KEY); // Clear if session becomes null
    }
  }, [sessionId]);

  // Function to convert frontend message format to backend API chat history format
  const formatChatHistoryForAPI = (history) => {
    return history.map(msg => ({
      role: msg.sender, // 'user' or 'ai'
      content: msg.text
    }));
  };

  const handleSendMessage = async (newMessageText) => {
    const userMessage = {
      id: Date.now().toString(), // Use timestamp for unique ID
      text: newMessageText,
      sender: 'user',
    };
    
    // Prepare messages for UI update (include user message immediately)
    const updatedMessagesForUI = [...messages, userMessage];
    setMessages(updatedMessagesForUI);
    setIsLoading(true);
    setError(null);

    const currentAiMessageId = Date.now().toString() + '-ai-stream';
    const aiMessagePlaceholder = {
      id: currentAiMessageId,
      text: '', // Start with empty text
      sender: 'ai',
      sources: [],
      isLoading: true, // Custom flag for the message itself
    };
    setMessages(prevMessages => [...prevMessages, aiMessagePlaceholder]);

    // Prepare only the current turn's messages if session_id exists, 
    // or full history if it's a new session (backend will handle populating memory)
    // The backend will primarily rely on its stored memory for the given session_id.
    // Sending history from client is mainly for new session bootstrapping or if session is lost.
    const historyForAPI = (sessionId && messages.length > 1) ? formatChatHistoryForAPI(messages) : []; 
    // If sessionId exists and there's prior history, send it. Otherwise, let backend use its memory or empty for new session.

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream' // Important to tell server we expect a stream
            },
            body: JSON.stringify({
                message: newMessageText,
                session_id: sessionId, // Send current session_id (can be null for first request)
                chat_history: historyForAPI // Send relevant history for potential new session memory population
            }),
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({ detail: 'Streaming API connection error' }));
            throw new Error(errData.detail || `API Error: ${response.status} ${response.statusText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let stillStreaming = true;

        while (stillStreaming) {
            const { value, done } = await reader.read();
            if (done) {
                stillStreaming = false;
                break;
            }

            const chunk = decoder.decode(value);
            // SSE messages are separated by \n\n. A single chunk might contain multiple messages.
            const eventLines = chunk.split('\n\n').filter(line => line.trim() !== '');

            for (const line of eventLines) {
                if (line.startsWith('data: ')) {
                    const jsonData = line.substring(6); // Remove "data: "
                    try {
                        const eventData = JSON.parse(jsonData);
                        if (eventData.type === 'session_id') {
                            console.log("Received session_id:", eventData.id);
                            setSessionId(eventData.id);
                        } else if (eventData.type === 'token') {
                            setMessages(prevMessages =>
                                prevMessages.map(msg =>
                                    msg.id === currentAiMessageId
                                        ? { ...msg, text: msg.text + eventData.content, isLoading: true }
                                        : msg
                                )
                            );
                        } else if (eventData.type === 'sources') {
                            setMessages(prevMessages =>
                                prevMessages.map(msg =>
                                    msg.id === currentAiMessageId
                                        ? { ...msg, sources: eventData.content }
                                        : msg
                                )
                            );
                        } else if (eventData.type === 'end') {
                            setMessages(prevMessages =>
                                prevMessages.map(msg =>
                                    msg.id === currentAiMessageId
                                        ? { ...msg, isLoading: false } // Mark as not loading
                                        : msg
                                )
                            );
                            stillStreaming = false;
                            break; 
                        } else if (eventData.type === 'error') {
                            console.error("Stream error:", eventData.content);
                            setMessages(prevMessages =>
                                prevMessages.map(msg =>
                                    msg.id === currentAiMessageId
                                        ? { ...msg, text: (msg.text || "") + `\\nStream Error: ${eventData.content}`, isLoading: false, isError: true }
                                        : msg
                                )
                            );
                            stillStreaming = false;
                            break;
                        }
                    } catch (e) {
                        console.error("Failed to parse stream event data:", jsonData, e);
                        setMessages(prevMessages =>
                            prevMessages.map(msg =>
                                msg.id === currentAiMessageId
                                    ? { ...msg, text: (msg.text || "") + `\\nError parsing stream data.`, isLoading: false, isError: true }
                                    : msg
                            )
                        );
                        stillStreaming = false;
                        break;
                    }
                }
            }
        }
    } catch (err) {
        console.error("Failed to send message or process stream:", err);
        setError(err.message);

        setMessages(prevMessages => {
            const aiMessageIndex = prevMessages.findIndex(m => m.id === currentAiMessageId);
            if (aiMessageIndex !== -1) {
                const currentMessage = prevMessages[aiMessageIndex];
                if (currentMessage.text === '') {
                    return prevMessages.filter(m => m.id !== currentAiMessageId);
                }
                return prevMessages.map((msg, idx) => 
                    idx === aiMessageIndex 
                        ? { ...msg, text: msg.text + `\\nRequest Error: ${err.message}`, isLoading: false, isError: true }
                        : msg
                );
            }
            return [
                ...prevMessages,
                { 
                    id: currentAiMessageId + '-fetch-error', 
                    text: `Error: ${err.message}`, 
                    sender: 'ai', 
                    isError: true, 
                    isLoading: false 
                }
            ];
        });
    } finally {
        setIsLoading(false); // Global loading indicator for input area
        // Final update to ensure isLoading is false on the specific message
        setMessages(prevMessages =>
            prevMessages.map(msg =>
                msg.id === currentAiMessageId ? { ...msg, isLoading: false } : msg
            )
        );
    }
  };

  // Button to clear session for testing
  const clearSession = () => {
    setSessionId(null);
    setMessages([
        { id: 'initial-ai-greeting-reset', text: 'Session cleared. Hello again!', sender: 'ai' },
    ]);
    localStorage.removeItem(SESSION_STORAGE_KEY);
    console.log("Session ID cleared and reset.");
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100 dark:bg-gray-900">
      <header className="bg-blue-600 text-white p-4 text-center shadow-md flex justify-between items-center">
        <h1 className="text-2xl font-semibold">UPSC RAG Helper</h1>
        <button 
          onClick={clearSession}
          className="bg-red-500 hover:bg-red-700 text-white font-bold py-1 px-3 rounded text-xs"
        >
          New Session
        </button>
      </header>
      {error && (
        <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 m-2 rounded-md" role="alert">
          <p className="font-bold">Error</p>
          <p>{error}</p>
        </div>
      )}
      <MessageDisplayArea messages={messages} isLoading={isLoading && messages.every(m => !m.isLoading)} /> 
      {/* Pass the global isLoading only if no message is individually loading */}
      <MessageInputArea onSendMessage={handleSendMessage} isLoading={isLoading} />
    </div>
  );
}

export default App; 