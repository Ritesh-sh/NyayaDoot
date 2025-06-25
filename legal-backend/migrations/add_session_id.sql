-- Add session_id column to chats table
ALTER TABLE chats ADD COLUMN session_id VARCHAR(255) NOT NULL DEFAULT 'default_session';

-- Create index for faster session lookups
CREATE INDEX idx_chats_session_id ON chats(session_id);

-- Update existing records to have unique session IDs
UPDATE chats SET session_id = CONCAT('session_', id) WHERE session_id = 'default_session'; 