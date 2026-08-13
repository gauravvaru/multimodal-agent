const ALLOWED_EXTENSIONS = new Set([
  ".pdf",
  ".jpg",
  ".jpeg",
  ".png",
  ".mp3",
  ".wav",
  ".m4a",
]);

const DEFAULT_MAX_UPLOAD_MB = 25;

export function getMaxUploadMb(): number {
  const configured = import.meta.env.VITE_MAX_UPLOAD_MB;
  if (!configured) {
    return DEFAULT_MAX_UPLOAD_MB;
  }
  const parsed = Number.parseInt(configured, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_MAX_UPLOAD_MB;
}

export function validateQuery(query: string): string[] {
  if (!query.trim()) {
    return ["Enter a query before running the agent."];
  }
  return [];
}

export function validateFiles(files: File[]): string[] {
  const errors: string[] = [];
  const maxBytes = getMaxUploadMb() * 1024 * 1024;

  files.forEach((file, index) => {
    const label = file.name || `file ${index + 1}`;
    const extension = getExtension(file.name);

    if (!ALLOWED_EXTENSIONS.has(extension)) {
      errors.push(
        `${label}: unsupported file type. Allowed: PDF, JPG, JPEG, PNG, MP3, WAV, M4A.`,
      );
    }

    if (file.size > maxBytes) {
      errors.push(`${label}: exceeds maximum size of ${getMaxUploadMb()} MB.`);
    }
  });

  return errors;
}

export function validateSubmission(query: string, files: File[]): string[] {
  return [...validateQuery(query), ...validateFiles(files)];
}

function getExtension(filename: string): string {
  const dotIndex = filename.lastIndexOf(".");
  if (dotIndex === -1) {
    return "";
  }
  return filename.slice(dotIndex).toLowerCase();
}
