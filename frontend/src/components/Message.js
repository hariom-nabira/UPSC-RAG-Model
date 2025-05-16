import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';

const UserAvatar = () => (
  <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white text-sm font-semibold mr-3 flex-shrink-0">
    U
  </div>
);

const AiAvatar = () => (
  <div className="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center text-white text-sm font-semibold mr-3 flex-shrink-0">
    AI
  </div>
);

const Message = ({ message }) => {
  const { text, sender, sources, isError } = message;
  const isUser = sender === 'user';

  const SourceLink = ({ source, index }) => {
    const { metadata, page_content } = source;
    const displayName = metadata?.display_name || metadata?.source_identifier || `Source ${index + 1}`;
    const pageLabel = metadata?.page_label || '';
    // const docUrl = metadata?.clickable_url || "#"; // If you have direct URLs

    return (
      <div className="mt-2 p-2 border border-gray-300 dark:border-gray-600 rounded-md bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 text-xs">
        <p className="font-semibold text-blue-600 dark:text-blue-400">{displayName} {pageLabel}</p>
        {page_content && (
            <p className="mt-1 text-gray-600 dark:text-gray-300 truncate">
                {page_content.substring(0,150)}{page_content.length > 150 ? '...' : ''}
            </p>
        )}
      </div>
    );
  };

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} my-1`}>
      <div className={`max-w-xl lg:max-w-2xl xl:max-w-3xl flex items-start ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        {!isUser && <AiAvatar />}
        {isUser && <UserAvatar />}
        <div
          className={`px-4 py-3 rounded-lg shadow-md prose prose-sm max-w-none 
            ${isError ? 'bg-red-400 text-white rounded-bl-none' :
              isUser
              ? 'bg-blue-500 text-white dark:bg-blue-600 dark:text-gray-100 rounded-br-none'
              : 'bg-white text-gray-800 dark:bg-gray-700 dark:text-gray-200 rounded-bl-none'
            }
          `}
        >
          <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
            {text}
          </ReactMarkdown>
          {sources && sources.length > 0 && !isUser && (
            <div className="mt-3 pt-2 border-t border-gray-300 dark:border-gray-600">
              <h4 className={`text-xs font-semibold mb-1 ${isUser ? 'text-blue-100' : 'text-gray-600 dark:text-gray-400'}`}>Sources:</h4>
              {sources.map((src, index) => (
                <SourceLink key={index} source={src} index={index} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Message; 