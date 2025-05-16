import React, { useEffect, useRef } from 'react';
import Message from './Message'; // Import the new Message component

const TypingIndicator = () => (
  <div className="flex items-center justify-start my-1">
    <div className="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center text-white text-sm font-semibold mr-3 flex-shrink-0">
      AI
    </div>
    <div className="px-4 py-3 rounded-lg shadow-md bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300">
      <div className="flex space-x-1 items-center">
        <span className="block w-2 h-2 bg-gray-500 dark:bg-gray-400 rounded-full animate-bounce_1"></span>
        <span className="block w-2 h-2 bg-gray-500 dark:bg-gray-400 rounded-full animate-bounce_2"></span>
        <span className="block w-2 h-2 bg-gray-500 dark:bg-gray-400 rounded-full animate-bounce_3"></span>
      </div>
    </div>
  </div>
);

const MessageDisplayArea = ({ messages, isLoading }) => {
  const endOfMessagesRef = useRef(null);

  const scrollToBottom = () => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]); // also scroll when loading indicator appears/disappears

  return (
    <div className="flex-grow p-4 md:p-6 overflow-y-auto space-y-2 bg-white dark:bg-gray-800">
      {messages.map((msg) => (
        <Message key={msg.id} message={msg} />
      ))}
      {isLoading && <TypingIndicator />}
      <div ref={endOfMessagesRef} />
    </div>
  );
};

export default MessageDisplayArea; 