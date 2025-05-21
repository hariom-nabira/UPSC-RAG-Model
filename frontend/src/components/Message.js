import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';

// Consistent Avatar Styling
const Avatar = ({ children, bgColor }) => (
  <div className={`w-8 h-8 rounded-full ${bgColor} flex items-center justify-center text-white text-xs font-semibold mr-3 flex-shrink-0 shadow`}>
    {children}
  </div>
);

const Message = ({ message }) => {
  const { text, sender, sources, isError, isLoading } = message; // Added isLoading for typing indicator styling
  const isUser = sender === 'user';

  const openSourceDocument = (sourcePath, pageNumber) => {
    if (!sourcePath) return;
    const fileName = sourcePath.split('/').pop();
    const pageInfo = pageNumber ? ` (page ${pageNumber})` : '';
    // const openConfirmed = window.confirm(`Opening source document: ${fileName}${pageInfo}`);
    // if (openConfirmed) { // Auto-open for now, confirmation can be added back if needed
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
      const apiBasePath = `${apiUrl}/api/v1/documents/`;
      let pageFragment = '';
      if (pageNumber) {
        pageFragment = `#page=${pageNumber}`;
      }
      const fullUrl = `${apiBasePath}${encodeURIComponent(sourcePath)}${pageFragment}`;
      try {
        window.open(fullUrl, '_blank');
      } catch (e) {
        console.error("Failed to open document:", e);
        alert(`Unable to open document. Path: ${sourcePath}`);
      }
    // }
  };

  const SourceLink = ({ source, index }) => {
    const { metadata, page_content } = source;
    const displayName = metadata?.file_name || metadata?.source_identifier || `Source ${index + 1}`;
    const pageLabelForDisplay = metadata?.page_label || (metadata?.page_number_str ? `Page: ${metadata.page_number_str}` : '');
    
    let sourcePath = '';
    if (metadata?.source) {
      if (metadata.source.startsWith('/') || metadata.source.startsWith('data/')) {
        sourcePath = metadata.source;
      } else {
        sourcePath = `data/${metadata.source}`;
      }
    }
    if (!sourcePath && metadata?.file_name) {
      sourcePath = `data/${metadata.file_name}`;
    }
    
    let pageNumberForUrl = null;
    if (metadata?.page !== undefined && metadata?.page !== null) {
        const parsedPage = parseInt(String(metadata.page));
        if (!isNaN(parsedPage)) {
            pageNumberForUrl = parsedPage;
        }
    }
        
    return (
      <button 
        onClick={() => sourcePath && openSourceDocument(sourcePath, pageNumberForUrl)}
        disabled={!sourcePath}
        title={sourcePath ? `Open: ${displayName}${pageNumberForUrl ? ` (page ${pageNumberForUrl})` : ''}` : 'Source file unavailable'}
        className={`mt-1.5 mr-1.5 px-2.5 py-1 text-xs rounded-full flex items-center 
                    bg-gray-100 hover:bg-gray-200 dark:bg-gray-700 dark:hover:bg-gray-600 
                    text-gray-700 dark:text-gray-200 
                    border border-gray-200 dark:border-gray-600 
                    disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-150`}
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5 mr-1.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V7a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        {displayName} {pageLabelForDisplay && `(${pageLabelForDisplay})`}
      </button>
    );
  };

  // Typing indicator for AI messages that are loading
  if (isLoading && !isUser) {
    return (
      <div className={`flex justify-start my-2`}>
        <div className={`flex items-start flex-row`}>
          <Avatar bgColor="bg-green-500">AI</Avatar>
          <div
            className={`px-4 py-3 rounded-2xl shadow-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 animate-pulse`}
          >
            <div className="flex space-x-1.5 items-center">
              <span className="block w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full"></span>
              <span className="block w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animation-delay-200ms"></span>
              <span className="block w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animation-delay-400ms"></span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Standard message display
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} my-2 animation-fadeIn`}>
      <div className={`max-w-xl lg:max-w-2xl xl:max-w-3xl flex items-start ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        {!isUser && <Avatar bgColor="bg-green-500">AI</Avatar>}
        {isUser && <Avatar bgColor="bg-blue-500">U</Avatar>}
        <div
          className={`px-4 py-2.5 rounded-2xl shadow-sm prose prose-sm dark:prose-invert max-w-none 
            ${isError ? 'bg-red-100 dark:bg-red-800 text-red-700 dark:text-red-200 border border-red-300 dark:border-red-600' :
              isUser
              ? 'bg-blue-500 text-white dark:bg-blue-600 dark:text-gray-50'
              : 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
            }
          `}
        >
          <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
            {text || (isUser ? '' : '...')}{/* Show ellipsis for empty AI message briefly before content streams */}
          </ReactMarkdown>
          {sources && sources.length > 0 && !isUser && (
            <div className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-600">
              {/* <h4 className={`text-xs font-semibold mb-1 ${isUser ? 'text-blue-100' : 'text-gray-500 dark:text-gray-400'}`}>Sources:</h4> */}
              <div className="flex flex-wrap">
                {sources.map((src, index) => (
                  <SourceLink key={index} source={src} index={index} />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Message; 