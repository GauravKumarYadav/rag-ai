-- Audit Logs Database Initialization Script
-- This script runs automatically when MySQL container starts for the first time

-- Create audit_logs table for tracking all API access
CREATE TABLE IF NOT EXISTS audit_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME(6) DEFAULT CURRENT_TIMESTAMP(6),
    user_id VARCHAR(255),
    username VARCHAR(255),
    client_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    resource VARCHAR(255),
    method VARCHAR(10),
    path VARCHAR(500),
    status_code INT,
    ip_address VARCHAR(45),
    user_agent TEXT,
    duration_ms FLOAT,
    request_summary TEXT,
    INDEX idx_user_id (user_id),
    INDEX idx_client_id (client_id),
    INDEX idx_timestamp (timestamp),
    INDEX idx_action (action),
    INDEX idx_status_code (status_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create users table for authentication
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_username (username),
    INDEX idx_email (email),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Seed initial admin user
-- Username: admin
-- Password: admin123 (bcrypt hashed)
-- IMPORTANT: Change this password immediately after first login!
INSERT INTO users (id, username, email, hashed_password, is_active, is_superuser)
VALUES (
    'a0000000-0000-0000-0000-000000000001',
    'admin',
    'admin@localhost',
    -- This is the bcrypt hash for 'admin123' with 12 rounds
    -- Generated with: import bcrypt; bcrypt.hashpw(b'admin123', bcrypt.gensalt(rounds=12))
    '$2b$12$amxpTmsjjIVl.wtqIKznhugQ5VOYar8vnP6C.bwc381d1yzLNpm42',
    TRUE,
    TRUE
)
ON DUPLICATE KEY UPDATE updated_at = CURRENT_TIMESTAMP;

-- Log that initialization is complete
SELECT 'Database initialization complete. Admin user created with username: admin, password: admin123' AS message;

-- ============================================================
-- Evaluation Framework Tables
-- ============================================================

-- Evaluation datasets - stores generated Q&A pairs for testing
CREATE TABLE IF NOT EXISTS evaluation_datasets (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    client_id VARCHAR(255),
    qa_pairs JSON NOT NULL,
    sample_size INT NOT NULL,
    active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_client_id (client_id),
    INDEX idx_active (active),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Evaluation runs - stores results of evaluation executions
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    dataset_id BIGINT NOT NULL,
    k_value INT DEFAULT 5,
    metrics JSON,
    detailed_results JSON,
    status ENUM('pending', 'running', 'completed', 'failed') DEFAULT 'pending',
    error_message TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    FOREIGN KEY (dataset_id) REFERENCES evaluation_datasets(id) ON DELETE CASCADE,
    INDEX idx_dataset_id (dataset_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================
-- Client Management Tables
-- ============================================================

-- Clients table - stores client entities for document organization
CREATE TABLE IF NOT EXISTS clients (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    aliases JSON,
    metadata JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_name (name),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
