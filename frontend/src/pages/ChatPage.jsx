import { useState, useRef, useEffect } from 'react';
import { useOutletContext } from 'react-router-dom';
import { Send, Sparkles, FileText, Clock, Trash2 } from 'lucide-react';
import { api } from '../lib/api';

export default function ChatPage() {
  const { online, checkHealth } = useOutletContext();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState('');
  const bottomRef = useRef(null);

  // Single health check on mount — no polling
  useEffect(() => { checkHealth(); }, []);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, streaming]);

  const send = async () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput('');

    setMessages(prev => [...prev, { role: 'user', content: q }]);
    setLoading(true);
    setStreaming('');

    try {
      let full = '';
      for await (const chunk of api.streamQuery(q)) {
        full += chunk;
        setStreaming(full);
      }
      const answer = full;
      setStreaming('');
      setMessages(prev => [...prev, { role: 'assistant', content: answer }]);
    } catch (err) {
      setStreaming('');
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${err.message}`, error: true }]);
    } finally {
      setLoading(false);
    }
  };

  const clear = () => { setMessages([]); setStreaming(''); };

  const handleKey = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } };

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)]">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Ask a question</h1>
        <p className="text-sm text-gray-500 mt-1">Search your document corpus with AI-powered RAG</p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto scrollbar-thin space-y-5 mb-4 pr-2">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <Sparkles size={40} className="mb-3 text-gray-300" />
            <p className="text-lg font-medium text-gray-500">Ask anything about your documents</p>
            <p className="text-sm mt-1">Your AI assistant has access to all indexed content</p>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`flex gap-3 ${m.role === 'user' ? 'justify-end' : ''}`}>
            {m.role === 'assistant' && (
              <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center flex-shrink-0 mt-0.5">
                <Sparkles size={14} className="text-white" />
              </div>
            )}
            <div className={`
              max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed
              ${m.role === 'user'
                ? 'bg-blue-600 text-white rounded-br-md'
                : m.error
                  ? 'bg-red-50 text-red-700 border border-red-200 rounded-bl-md'
                  : 'bg-white border border-gray-200 text-gray-800 rounded-bl-md shadow-sm'
              }
            `}>
              <div className="whitespace-pre-wrap">{m.content}</div>
              {m.role === 'assistant' && !m.error && (
                <div className="flex items-center gap-3 mt-2 pt-2 border-t border-gray-100 text-[10px] text-gray-400">
                  <span className="flex items-center gap-1"><FileText size={10} /> Sources from knowledge base</span>
                </div>
              )}
            </div>
            {m.role === 'user' && (
              <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="text-white text-xs font-bold">U</span>
              </div>
            )}
          </div>
        ))}

        {/* Streaming indicator */}
        {loading && streaming && (
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center flex-shrink-0">
              <Sparkles size={14} className="text-white" />
            </div>
            <div className="max-w-[80%] rounded-2xl rounded-bl-md px-4 py-3 text-sm leading-relaxed bg-white border border-blue-200 text-gray-800 shadow-sm">
              <div className="whitespace-pre-wrap">{streaming}</div>
              <span className="inline-block w-2 h-4 bg-blue-600 animate-pulse ml-0.5 align-text-bottom" />
            </div>
          </div>
        )}

        {loading && !streaming && (
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center flex-shrink-0">
              <Sparkles size={14} className="text-white" />
            </div>
            <div className="bg-white border border-gray-200 rounded-2xl rounded-bl-md px-4 py-3 shadow-sm">
              <div className="flex gap-1.5">
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-2 flex gap-2">
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Type your question and press Enter..."
          rows={1}
          disabled={!online}
          className="flex-1 resize-none bg-transparent px-3 py-2.5 text-sm outline-none placeholder:text-gray-400 disabled:opacity-50"
        />
        <div className="flex items-center gap-1 pr-1">
          {messages.length > 0 && (
            <button onClick={clear} className="p-2 rounded-lg hover:bg-gray-100 text-gray-400 hover:text-gray-600 transition-colors" title="Clear chat">
              <Trash2 size={16} />
            </button>
          )}
          <button
            onClick={send}
            disabled={!input.trim() || loading || !online}
            className="p-2.5 rounded-xl bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            <Send size={16} />
          </button>
        </div>
      </div>

      {!online && (
        <p className="text-center text-xs text-red-500 mt-2">Backend is offline. Start the server to begin chatting.</p>
      )}
    </div>
  );
}
