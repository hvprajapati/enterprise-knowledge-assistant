import { useState, useRef } from 'react';
import { UploadCloud, File, CheckCircle, AlertCircle, X } from 'lucide-react';
import { api } from '../lib/api';

export default function UploadPage() {
  const [dragOver, setDragOver] = useState(false);
  const [selected, setSelected] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const inputRef = useRef(null);

  const handleFile = async (file) => {
    if (!file) return;
    setSelected(file);
    setResult(null);
    setUploading(true);
    try {
      const data = await api.upload(file);
      setResult({ ok: true, msg: `"${data.filename}" uploaded & indexed — job ${data.job_id.slice(0, 8)}...` });
      setSelected(null);
    } catch (err) {
      setResult({ ok: false, msg: err.message });
    } finally {
      setUploading(false);
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const accepted = ['.pdf', '.docx', '.txt', '.md', 'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'text/plain', 'text/markdown', 'text/x-markdown'];

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Upload documents</h1>
        <p className="text-sm text-gray-500 mt-1">Add PDF, DOCX, TXT, or Markdown files to your knowledge base</p>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`
          relative border-2 border-dashed rounded-2xl p-14 text-center cursor-pointer
          transition-all duration-200
          ${dragOver
            ? 'border-blue-400 bg-blue-50/50'
            : 'border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50/50'
          }
          ${uploading ? 'pointer-events-none opacity-60' : ''}
        `}
      >
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          accept={accepted.join(',')}
          onChange={e => handleFile(e.target.files[0])}
        />

        {selected ? (
          <div className="flex flex-col items-center gap-3">
            <File size={40} className="text-blue-600" />
            <div>
              <p className="font-semibold text-gray-900">{selected.name}</p>
              <p className="text-sm text-gray-500">{(selected.size / 1024 / 1024).toFixed(1)} MB</p>
            </div>
            <button
              onClick={e => { e.stopPropagation(); setSelected(null); }}
              className="mt-1 px-3 py-1 text-xs rounded-full bg-gray-100 hover:bg-gray-200 text-gray-600 transition-colors"
            >
              Cancel
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3">
            <div className="w-14 h-14 rounded-2xl bg-blue-50 flex items-center justify-center">
              <UploadCloud size={24} className="text-blue-600" />
            </div>
            <div>
              <p className="font-semibold text-gray-900">Drop your file here or click to browse</p>
              <p className="text-sm text-gray-500 mt-0.5">PDF · DOCX · TXT · Markdown — up to 50 MB</p>
            </div>
          </div>
        )}
      </div>

      {/* Upload status */}
      {uploading && (
        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-xl flex items-center gap-3">
          <div className="w-5 h-5 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-blue-700">Uploading and indexing...</p>
        </div>
      )}

      {result && (
        <div className={`mt-4 p-4 rounded-xl flex items-start gap-3 ${
          result.ok ? 'bg-emerald-50 border border-emerald-200' : 'bg-red-50 border border-red-200'
        }`}>
          {result.ok
            ? <CheckCircle size={18} className="text-emerald-600 mt-0.5 flex-shrink-0" />
            : <AlertCircle size={18} className="text-red-600 mt-0.5 flex-shrink-0" />
          }
          <div className="flex-1">
            <p className={`text-sm font-medium ${result.ok ? 'text-emerald-800' : 'text-red-800'}`}>
              {result.ok ? 'Upload successful' : 'Upload failed'}
            </p>
            <p className={`text-sm mt-0.5 ${result.ok ? 'text-emerald-700' : 'text-red-700'}`}>
              {result.msg}
            </p>
          </div>
          <button onClick={() => setResult(null)} className="text-gray-400 hover:text-gray-600">
            <X size={16} />
          </button>
        </div>
      )}

      {/* Info cards */}
      <div className="grid grid-cols-3 gap-4 mt-8">
        {[
          { label: 'PDF', desc: 'Full text extraction with page-level error handling', color: 'red' },
          { label: 'DOCX', desc: 'Word documents with structure preservation', color: 'blue' },
          { label: 'TXT / MD', desc: 'Plain text with encoding auto-detection', color: 'gray' },
        ].map(({ label, desc, color }) => (
          <div key={label} className="bg-white border border-gray-200 rounded-xl p-4">
            <span className={`inline-block text-[11px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-${color}-50 text-${color}-600 mb-2`}>
              {label}
            </span>
            <p className="text-xs text-gray-500 mt-1">{desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
