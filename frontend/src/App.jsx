import { useEffect, useState } from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import { api } from './lib/api';

export default function App() {
  const [online, setOnline] = useState(false);
  const [fileCount, setFileCount] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const refresh = async () => {
    try {
      await api.health();
      setOnline(true);
    } catch {
      setOnline(false);
    }
  };

  // Health: every 30s (1 request). File count: every 2 min (1 request).
  useEffect(() => {
    refresh();
    const healthId = setInterval(refresh, 30000);
    const loadFiles = async () => {
      try { const f = await api.listFiles(); setFileCount(Array.isArray(f) ? f.length : 0); } catch {}
    };
    loadFiles();
    const filesId = setInterval(loadFiles, 120000);
    return () => { clearInterval(healthId); clearInterval(filesId); };
  }, []);

  return (
    <div className="flex h-screen overflow-hidden bg-gray-50">
      <Sidebar
        online={online}
        fileCount={fileCount}
        collapsed={!sidebarOpen}
        onToggle={() => setSidebarOpen(v => !v)}
      />
      <main className="flex-1 overflow-y-auto scrollbar-thin">
        <div className="max-w-5xl mx-auto px-6 py-8">
          <Outlet context={{ refresh, online }} />
        </div>
      </main>
    </div>
  );
}
