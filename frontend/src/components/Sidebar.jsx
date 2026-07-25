import { NavLink, useLocation } from 'react-router-dom';
import { MessageSquare, Upload, FolderOpen, Terminal, ChevronLeft, ChevronRight, Database } from 'lucide-react';

const nav = [
  { to: '/', icon: MessageSquare, label: 'Chat' },
  { to: '/upload', icon: Upload, label: 'Upload' },
  { to: '/documents', icon: FolderOpen, label: 'Documents' },
  { to: '/console', icon: Terminal, label: 'Console' },
];

export default function Sidebar({ online, fileCount, collapsed, onToggle }) {
  const loc = useLocation();

  return (
    <aside className={`
      flex flex-col bg-gray-900 text-gray-300 transition-all duration-200
      ${collapsed ? 'w-[72px]' : 'w-[260px]'}
    `}>
      {/* Brand */}
      <div className={`
        flex items-center border-b border-gray-800 px-5
        ${collapsed ? 'justify-center h-16' : 'justify-between h-16'}
      `}>
        {!collapsed && (
          <div>
            <div className="text-sm font-bold text-white tracking-tight">Knowledge Assistant</div>
            <div className="text-[10px] text-gray-500 uppercase tracking-wider">Enterprise RAG</div>
          </div>
        )}
        {collapsed && <Database size={22} className="text-blue-400" />}
        <button
          onClick={onToggle}
          className="p-1.5 rounded-lg hover:bg-gray-800 text-gray-500 hover:text-gray-300 transition-colors"
        >
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-4 px-3 space-y-0.5">
        {nav.map(({ to, icon: Icon, label }) => {
          const active = to === '/' ? loc.pathname === '/' : loc.pathname.startsWith(to);
          return (
            <NavLink
              key={to}
              to={to}
              className={`
                flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium
                transition-colors duration-150
                ${active
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/25'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
                }
                ${collapsed ? 'justify-center px-2' : ''}
              `}
              title={collapsed ? label : undefined}
            >
              <Icon size={19} />
              {!collapsed && <span>{label}</span>}
            </NavLink>
          );
        })}
      </nav>

      {/* Status footer */}
      <div className={`
        border-t border-gray-800 p-4
        ${collapsed ? 'text-center' : ''}
      `}>
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <span className={`w-2 h-2 rounded-full ${online ? 'bg-emerald-400 shadow-md shadow-emerald-400/50' : 'bg-red-400'}`} />
          {!collapsed && <span>{online ? 'Backend online' : 'Backend offline'}</span>}
        </div>
        {!collapsed && fileCount > 0 && (
          <div className="text-[11px] text-gray-600 mt-1 ml-4">{fileCount} document{fileCount !== 1 ? 's' : ''} indexed</div>
        )}
      </div>
    </aside>
  );
}
