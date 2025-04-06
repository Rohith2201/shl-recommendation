const config = {
  apiUrl: process.env.NODE_ENV === 'production'
    ? 'https://shl-assessment-api.onrender.com'
    : 'http://localhost:8000'
};

export default config; 