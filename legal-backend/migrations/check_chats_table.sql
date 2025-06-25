-- Check if chats table exists
CREATE TABLE IF NOT EXISTS chats (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    message TEXT NOT NULL,
    response TEXT NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Check if session_id column exists
SELECT COUNT(*) INTO @exists 
FROM information_schema.columns 
WHERE table_schema = DATABASE()
AND table_name = 'chats' 
AND column_name = 'session_id';

-- Add session_id column if it doesn't exist
SET @sql = IF(@exists = 0,
    'ALTER TABLE chats ADD COLUMN session_id VARCHAR(255) NOT NULL DEFAULT "default_session"',
    'SELECT "session_id column already exists"'
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Create index for faster session lookups if it doesn't exist
CREATE INDEX IF NOT EXISTS idx_chats_session_id ON chats(session_id);

-- Update existing records to have unique session IDs if they have default_session
UPDATE chats SET session_id = CONCAT('session_', id) WHERE session_id = 'default_session'; 