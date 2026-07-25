import { useState } from 'react';
import { Play, Search, CheckCircle, AlertCircle, Clock, Wrench, Server, Cpu, HardDrive } from 'lucide-react';
import { api } from '../lib/api';
import { useOutletContext } from 'react-router-dom';

export default function ConsolePage() {
  const { online, checkHealth, loadFileCount } = useOutletContext();

  // Index
  const [directory, setDirectory] = useState('data/uploads');
  const [indexResult, setIndexResult] = useState(null);
  const [indexing, setIndexing] = useState(false);

  // Job check
  const [jobId, setJobId] = useState('');
  const [jobResult, setJobResult] = useState(null);

  // System info
  const [sysInfo, setSysInfo] = useState(null);

  const doIndex = async () => {
    setIndexing(true);
    setIndexResult(null);
    try {
      const data = await api.indexDirectory(directory);
      setIndexResult({ ok: true, msg: `Indexing started — job ${data.job_id.slice(0, 8)}... status: ${data.status}` });
    } catch (err) {
      setIndexResult({ ok: false, msg: err.message });
    } finally {
      setIndexing(false);
    }
  };

  const checkJob = async () => {
    try {
      const data = await api.getJob(jobId);
      setJobResult({
        status: data.status,
        files: data.files_processed || 0,
        chunks: data.chunks_created || 0,
        embeddings: data.embeddings_generated || 0,
        error: data.error_message,
      });
    } catch (err) {
      setJobResult({ error: err.message });
    }
  };

  const loadSys = async () => {
    try {
      await checkHealth();
      await loadFileCount();
      const [health, files] = await Promise.all([
        api.health(),
        api.listFiles(),
      ]);
      setSysInfo({
        health: health.status || 'unknown',
        files: Array.isArray(files) ? files.length : 0,
        provider: 'deepseek',
        uptime: '—',
      });
    } catch (err) {
      setSysInfo({ error: err.message });
    }
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Console</h1>
        <p className="text-sm text-gray-500 mt-1">Index management, job tracking, and system status</p>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Index directory */}
        <div className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-8 h-8 rounded-lg bg-blue-50 flex items-center justify-center">
              <Play size={15} className="text-blue-600" />
            </div>
            <h2 className="font-semibold text-gray-900">Index directory</h2>
          </div>

          <label className="text-xs font-medium text-gray-500 uppercase tracking-wider">Path</label>
          <input
            type="text"
            value={directory}
            onChange={e => setDirectory(e.target.value)}
            className="w-full mt-1 px-3 py-2.5 text-sm bg-gray-50 border border-gray-200 rounded-xl outline-none focus:border-blue-400 focus:ring-3 focus:ring-blue-50 transition-all font-mono"
          />

          <button
            onClick={doIndex}
            disabled={indexing || !online}
            className="mt-4 w-full flex items-center justify-center gap-2 px-4 py-2.5 text-sm font-medium rounded-xl bg-gray-900 text-white hover:bg-gray-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {indexing ? (
              <><div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" /> Indexing...</>
            ) : (
              <><Play size={14} /> Start indexing</>
            )}
          </button>

          {indexResult && (
            <div className={`mt-4 p-3 rounded-xl text-sm flex items-start gap-2 ${
              indexResult.ok ? 'bg-emerald-50 text-emerald-800' : 'bg-red-50 text-red-800'
            }`}>
              {indexResult.ok ? <CheckCircle size={15} className="mt-0.5 flex-shrink-0" /> : <AlertCircle size={15} className="mt-0.5 flex-shrink-0" />}
              {indexResult.msg}
            </div>
          )}
        </div>

        {/* Check job */}
        <div className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-8 h-8 rounded-lg bg-amber-50 flex items-center justify-center">
              <Search size={15} className="text-amber-600" />
            </div>
            <h2 className="font-semibold text-gray-900">Job status</h2>
          </div>

          <label className="text-xs font-medium text-gray-500 uppercase tracking-wider">Job ID</label>
          <input
            type="text"
            value={jobId}
            onChange={e => setJobId(e.target.value)}
            placeholder="Paste job ID..."
            className="w-full mt-1 px-3 py-2.5 text-sm bg-gray-50 border border-gray-200 rounded-xl outline-none focus:border-blue-400 font-mono"
          />

          <button
            onClick={checkJob}
            disabled={!jobId.trim()}
            className="mt-4 w-full flex items-center justify-center gap-2 px-4 py-2.5 text-sm font-medium rounded-xl bg-white border border-gray-200 hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            <Search size={14} /> Check status
          </button>

          {jobResult && (
            <div className="mt-4 space-y-2">
              {jobResult.error ? (
                <div className="p-3 bg-red-50 text-red-800 text-sm rounded-xl">{jobResult.error}</div>
              ) : (
                <>
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${jobResult.status === 'COMPLETED' ? 'bg-emerald-500' : jobResult.status === 'FAILED' ? 'bg-red-500' : 'bg-amber-500 animate-pulse'}`} />
                    <span className="text-sm font-medium text-gray-700">{jobResult.status}</span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-center">
                    {[
                      { label: 'Files', value: jobResult.files },
                      { label: 'Chunks', value: jobResult.chunks },
                      { label: 'Embeddings', value: jobResult.embeddings },
                    ].map(({ label, value }) => (
                      <div key={label} className="bg-gray-50 rounded-xl p-2">
                        <div className="text-lg font-bold text-gray-900">{value}</div>
                        <div className="text-[10px] text-gray-500 uppercase">{label}</div>
                      </div>
                    ))}
                  </div>
                  {jobResult.error && (
                    <p className="text-xs text-red-600">{jobResult.error}</p>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      </div>

      {/* System info */}
      <div className="mt-6 bg-white border border-gray-200 rounded-2xl p-6 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gray-100 flex items-center justify-center">
              <Server size={15} className="text-gray-600" />
            </div>
            <h2 className="font-semibold text-gray-900">System</h2>
          </div>
          <button
            onClick={loadSys}
            className="text-xs font-medium text-blue-600 hover:text-blue-700 transition-colors"
          >
            Refresh
          </button>
        </div>

        {sysInfo ? (
          <div className="grid grid-cols-4 gap-4">
            {[
              { icon: Server, label: 'Backend', value: sysInfo.health || '—', color: sysInfo.health === 'healthy' ? 'emerald' : 'red' },
              { icon: HardDrive, label: 'Documents', value: `${sysInfo.files || 0} files`, color: 'blue' },
              { icon: Cpu, label: 'LLM Provider', value: sysInfo.provider || '—', color: 'gray' },
              { icon: Clock, label: 'Uptime', value: sysInfo.uptime || '—', color: 'gray' },
            ].map(({ icon: Icon, label, value, color }) => (
              <div key={label} className="bg-gray-50 rounded-xl p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Icon size={14} className={`text-${color}-500`} />
                  <span className="text-[11px] font-medium text-gray-500 uppercase tracking-wider">{label}</span>
                </div>
                <div className={`text-sm font-semibold text-${color}-700`}>{value}</div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-sm text-gray-400 text-center py-8">
            Click "Refresh" to load system information
          </div>
        )}

        {sysInfo?.error && (
          <div className="mt-2 p-3 bg-red-50 text-red-700 text-sm rounded-xl">{sysInfo.error}</div>
        )}
      </div>
    </div>
  );
}
