# Architecture Overview

## System Design
┌─────────────┐
│ Student │
│ Interface │
└──────┬──────┘
│
▼
┌─────────────┐ ┌──────────┐
│ Flask │─────▶│ Groq │
│ Backend │ │ API │
└──────┬──────┘ └──────────┘
│
▼
┌─────────────┐
│ Supabase │
│ (Auth+DB) │
└─────────────┘

## Key Components

**`get_answer()`**: Main AI interaction logic
- Builds conversation context
- Calls Groq with teaching prompt
- Handles response formatting

**`get_teaching_prompt()`**: Prompt engineering
- 3 modes: socratic, direct, guided
- Max 2 verification questions
- Adapts to multi-part concepts

**Teacher Dashboard**: 
- Real-time student activity
- AI-generated summaries
- Struggle detection alerts

## Database Schema
- `students`: User accounts
- `classrooms`: Teacher-created classes
- `enrollments`: Student-class relationships
- `messages`: Chat history
- `student_profiles`: Learning patterns
- `struggle_events`: Difficulty tracking
- `teacher_alerts`: Auto-generated notifications
