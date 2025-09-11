const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8888';

export const API_ENDPOINTS = {
  PREDICT: `${API_BASE_URL}/predict`,
};
