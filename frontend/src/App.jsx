import { useState, useCallback } from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import { api } from './lib/api';

export default function App() {
  const [online, setOnline] = useState(null); // null = not checked yet
  const [fileCount, setFileCount] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Only check health when user explicitly triggers it — no polling
  const checkHealth = useCallback(async () => {
    try {
      await api.health();
      setOnline(true);
    } catch {
      setOnline(false);
    }
  }, []);

  const loadFileCount = useCallback(async () => {
    try {
      const f = await api.listFiles();
      setFileCount(Array.isArray(f) ? f.length : 0);
    } catch {}
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
          <Outlet context={{ checkHealth, loadFileCount, online, setOnline }} />
        </div>
      </main>
    </div>
  );
}
