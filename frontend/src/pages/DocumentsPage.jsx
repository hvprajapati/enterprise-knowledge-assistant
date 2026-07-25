import { useState, useEffect } from 'react';
import { File, Trash2, RefreshCw, HardDrive, Search } from 'lucide-react';
import { api } from '../lib/api';

function fmtSize(bytes) {
  for (const unit of ['B', 'KB', 'MB', 'GB']) {
    if (bytes < 1024) return `${bytes.toFixed(1)} ${unit}`;
    bytes /= 1024;
  }
  return `${bytes.toFixed(1)} TB`;
}

function fmtDate(iso) {
  try { return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' }); }
  catch { return iso?.slice(0, 19) || '—'; }
}

export default function DocumentsPage() {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [deleteTarget, setDeleteTarget] = useState('');
  const [deleteStatus, setDeleteStatus] = useState('');
  const [search, setSearch] = useState('');

  const load = async () => {
    setLoading(true);
    try {
      const data = await api.listFiles();
      setFiles(Array.isArray(data) ? data : []);
    } catch { setFiles([]); }
    finally { setLoading(false); }
  };

  useEffect(() => { load(); }, []);

  const del = async () => {
    if (!deleteTarget.trim()) return;
    try {
      const res = await api.deleteFile(deleteTarget.trim());
      setDeleteStatus(`Deleted: ${deleteTarget}`);
      setDeleteTarget('');
      load();
    } catch (err) {
      setDeleteStatus(`Error: ${err.message}`);
    }
  };

  const filtered = files.filter(f =>
    f.filename.toLowerCase().includes(search.toLowerCase())
  );

  const totalSize = files.reduce((s, f) => s + (f.size || 0), 0);

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Documents</h1>
          <p className="text-sm text-gray-500 mt-1">{files.length} file{files.length !== 1 ? 's' : ''} indexed · {fmtSize(totalSize)} total</p>
        </div>
        <button
          onClick={load}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-xl bg-white border border-gray-200 hover:bg-gray-50 transition-colors disabled:opacity-50"
        >
          <RefreshCw size={15} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* Search */}
      <div className="relative mb-6">
        <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
        <input
          type="text"
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Filter files..."
          className="w-full pl-9 pr-4 py-2.5 text-sm bg-white border border-gray-200 rounded-xl outline-none focus:border-blue-400 focus:ring-3 focus:ring-blue-50 transition-all"
        />
      </div>

      {/* File list */}
      <div className="bg-white border border-gray-200 rounded-2xl overflow-hidden shadow-sm">
        {loading ? (
          <div className="flex items-center justify-center py-16 text-gray-400">
            <RefreshCw size={20} className="animate-spin mr-2" />
            <span className="text-sm">Loading...</span>
          </div>
        ) : filtered.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-gray-400">
            <HardDrive size={36} className="mb-3 text-gray-300" />
            <p className="text-sm font-medium text-gray-500">
              {search ? 'No files match your search' : 'No documents uploaded yet'}
            </p>
            <p className="text-xs text-gray-400 mt-1">
              {search ? 'Try a different search term' : 'Upload your first document to get started'}
            </p>
          </div>
        ) : (
          <div>
            {filtered.map((f, i) => (
              <div key={f.filename + i} className="flex items-center justify-between px-5 py-3.5 border-b border-gray-100 last:border-b-0 hover:bg-gray-50/50 transition-colors">
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-9 h-9 rounded-lg bg-blue-50 flex items-center justify-center flex-shrink-0">
                    <File size={16} className="text-blue-600" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">{f.filename}</p>
                    <p className="text-xs text-gray-400 mt-0.5">
                      {fmtSize(f.size || 0)} · {fmtDate(f.uploaded_at)}
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => { setDeleteTarget(f.filename); setDeleteStatus(''); }}
                  className="p-2 rounded-lg hover:bg-red-50 text-gray-400 hover:text-red-600 transition-colors flex-shrink-0"
                  title="Delete file"
                >
                  <Trash2 size={15} />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Delete confirmation */}
      {deleteTarget && (
        <div className="mt-6 p-5 bg-red-50 border border-red-200 rounded-2xl">
          <p className="text-sm font-medium text-red-800 mb-2">
            Delete "{deleteTarget}"?
          </p>
          <p className="text-xs text-red-600 mb-4">
            This removes the file from disk. Vectors in FAISS/SQLite are not automatically cleaned up.
          </p>
          <div className="flex gap-2">
            <button
              onClick={del}
              className="px-4 py-2 text-sm font-medium rounded-xl bg-red-600 text-white hover:bg-red-700 transition-colors"
            >
              Yes, delete
            </button>
            <button
              onClick={() => setDeleteTarget('')}
              className="px-4 py-2 text-sm font-medium rounded-xl bg-white border border-gray-200 hover:bg-gray-50 transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {deleteStatus && !deleteTarget && (
        <p className="mt-4 text-sm text-gray-600">{deleteStatus}</p>
      )}
    </div>
  );
}
