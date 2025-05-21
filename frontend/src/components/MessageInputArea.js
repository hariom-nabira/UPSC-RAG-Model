import React, { useState, useEffect, useRef } from 'react';

const MessageInputArea = ({ onSendMessage, isLoading }) => {
  const [inputText, setInputText] = useState('');
  const textareaRef = useRef(null);

  // Auto-resize textarea height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'; // Reset height
      const scrollHeight = textareaRef.current.scrollHeight;
      // Max height for textarea, e.g., 5 lines (approx 20px per line + padding)
      const maxHeight = 5 * 20 + 2 * 12; // Assuming 1.5rem line height (24px) and py-2.5 (10px * 2)
      textareaRef.current.style.height = `${Math.min(scrollHeight, maxHeight)}px`;
    }
  }, [inputText]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputText.trim() && !isLoading) {
      onSendMessage(inputText.trim());
      setInputText('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 p-3 md:p-4 border-t border-gray-200 dark:border-gray-700 sticky bottom-0">
      <form onSubmit={handleSubmit} className="flex items-end space-x-2">
        <textarea
          ref={textareaRef}
          className="flex-grow p-2.5 pr-10 border border-gray-300 dark:border-gray-600 rounded-xl resize-none 
                     bg-gray-50 dark:bg-gray-700 text-gray-800 dark:text-gray-200 
                     focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none 
                     transition-all duration-150 leading-relaxed scrollbar-thin"
          rows="1"
          placeholder="Send a message..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
          style={{ maxHeight: '120px' }} // Corresponds to roughly 5 lines
        />
        <button
          type="submit"
          className={`p-2.5 rounded-xl text-white 
                      bg-indigo-600 hover:bg-indigo-700 
                      dark:bg-indigo-500 dark:hover:bg-indigo-600 
                      focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 
                      dark:focus:ring-offset-gray-800
                      transition-colors duration-150 
                      disabled:opacity-60 disabled:cursor-not-allowed 
                      flex items-center justify-center shadow-sm`}
          disabled={!inputText.trim() || isLoading}
          title="Send message"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
            {/* Paper airplane icon: */}
            <path d="M3.105 3.105a1.5 1.5 0 012.122-.001l11.06 7.402a1.5 1.5 0 010 2.492L5.227 20.4a1.5 1.5 0 01-2.122-2.122L14.343 12 5.227 5.722a1.5 1.5 0 01-.001-2.122z" />
          </svg>
        </button>
      </form>
      {/* Quick replies placeholder could go here */}
    </div>
  );
};

export default MessageInputArea;
