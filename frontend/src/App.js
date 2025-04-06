import React, { useState } from 'react';
import {
  Container,
  Typography,
  TextField,
  Button,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Box,
  CircularProgress,
  Alert,
} from '@mui/material';
import axios from 'axios';

const API_URL = 'http://localhost:8000';

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await axios.get(`${API_URL}/api/recommend`, {
        params: { query }
      });
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom align="center">
        SHL Assessment Recommender
      </Typography>
      
      <Paper component="form" onSubmit={handleSubmit} sx={{ p: 3, mb: 4 }}>
        <TextField
          fullWidth
          label="Enter job description or requirements"
          multiline
          rows={4}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          sx={{ mb: 2 }}
        />
        <Button
          variant="contained"
          type="submit"
          disabled={loading || !query.trim()}
          fullWidth
        >
          {loading ? <CircularProgress size={24} /> : 'Get Recommendations'}
        </Button>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 4 }}>
          {error}
        </Alert>
      )}

      {results && (
        <>
          <Box sx={{ mb: 4 }}>
            <Typography variant="h6" gutterBottom>
              Explanation
            </Typography>
            <Paper sx={{ p: 2 }}>
              <Typography>{results.explanation}</Typography>
            </Paper>
          </Box>

          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Assessment Name</TableCell>
                  <TableCell>Duration</TableCell>
                  <TableCell>Test Type</TableCell>
                  <TableCell>Remote Testing</TableCell>
                  <TableCell>Adaptive/IRT</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {results.recommendations.map((assessment) => (
                  <TableRow key={assessment.name}>
                    <TableCell>
                      <a href={assessment.url} target="_blank" rel="noopener noreferrer">
                        {assessment.name}
                      </a>
                    </TableCell>
                    <TableCell>{assessment.duration}</TableCell>
                    <TableCell>{assessment.test_type}</TableCell>
                    <TableCell>{assessment.remote_testing ? 'Yes' : 'No'}</TableCell>
                    <TableCell>{assessment.adaptive_irt ? 'Yes' : 'No'}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </>
      )}
    </Container>
  );
}

export default App; 