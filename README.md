# PodcastAI - Audio Intelligence Platform

A web-based application that transforms podcast audio into searchable transcripts and provides AI-powered question answering capabilities.

## Overview

PodcastAI enables users to upload audio files or provide podcast URLs for automatic transcription and intelligent content analysis. The platform uses advanced speech-to-text technology to convert audio content into searchable text, allowing users to quickly find specific information through natural language queries.

## Features

- **Audio Upload**: Support for multiple audio formats (MP3, WAV, M4A)
- **URL Processing**: Direct transcription from audio URLs
- **Automatic Transcription**: Speech-to-text conversion using OpenAI Whisper
- **Question Answering**: AI-powered responses based on transcript content
- **Topic Extraction**: Automatic generation of content tags and keywords
- **Responsive Design**: Professional interface optimized for all devices

## How It Works

1. Users can either upload audio files directly or provide a URL to an audio resource
2. The system processes the audio using OpenAI's Whisper API to generate accurate text transcripts
3. Transcripts are analyzed and vectorized using TF-IDF for efficient searching
4. User queries are matched against the transcript content using cosine similarity
5. The most relevant content sections are returned as answers

## Technical Architecture

### Backend

- **Framework**: FastAPI (Python)
- **Audio Processing**: OpenAI Whisper API
- **Text Analysis**: scikit-learn (TF-IDF vectorization)
- **Similarity Matching**: Cosine similarity algorithms
- **File Handling**: Python multipart for file uploads
- **HTTP Requests**: Requests library for URL processing

### Frontend

- **HTML5**: Semantic markup and structure
- **CSS3**: Modern styling with gradients and animations
- **JavaScript**: Interactive functionality and API communication
- **Three.js**: Subtle 3D background effects
- **Responsive Design**: Mobile-first approach

### Key Libraries

```
- FastAPI
- OpenAI Python SDK
- scikit-learn
- requests
- python-multipart
- Three.js (frontend)
```

## Usage

### Upload Audio File

1. Click on the upload area or drag and drop an audio file
2. Click "Transcribe" to process the audio
3. Wait for transcription to complete

### Process Audio URL

1. Paste a direct link to an audio file in the URL field
2. Click "Process URL" to download and transcribe
3. Review the generated transcript

### Ask Questions

1. Once transcript is available, use the Q&A section
2. Type your question about the content
3. Receive AI-generated answers based on the transcript

## Future Enhancements

- Database integration for transcript persistence
- Advanced NLP models for improved question answering
- Speaker identification and diarization
- Real-time audio processing
- User authentication and session management
- Batch processing capabilities

## License

All rights reserved.

## Support

For issues and questions, please create an issue in the repository or contact the development team.
