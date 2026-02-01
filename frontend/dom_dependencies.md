# DOM Dependencies & Business Features

This file documents the critical DOM elements (IDs and Classes) that the JavaScript logic relies on.
**CRITICAL**: When refactoring HTML, ensure these IDs are preserved or the JS is updated accordingly.

## 1. `frontend/index.html` (Main Application)

### Authentication & User
- `#logoutBtn`: Button to trigger logout.
- `#userName`: Element displaying current username.
- `#userAvatar`: Element displaying user avatar/initial.
- `#profileBtn`: Button to navigate to profile page.

### Sidebar & Navigation
- `#newChatBtn`: Button to start a new conversation.
- `#conversationList`: Container for conversation items.
- `.conversation-item`: Class for individual conversation items (created dynamically).
- `.delete-btn`: Class for delete button within conversation item.

### Chat Interface
- `#chatTitle`: Header displaying conversation title.
- `#modelBadge`: Header displaying model name.
- `#chatClientSelect`: Dropdown in header for client context.
- `#statusDot`, `#statusText`: Connection status indicators.
- `#chatMessages`: Container for chat history and new messages.
- `#welcomeState`: Container shown when no messages exist.
- `#typingIndicator`: Container for "AI is thinking..." animation.
- `#chatInput`: Textarea for user input.
- `#sendBtn`: Button to submit message.

### Documents Panel
- `#docsBtn`: Button in header to open documents panel.
- `#documentsPanel`: The side panel container.
- `#closePanelBtn`: Button to close the panel.
- `#clientSelect`: Dropdown in panel (synced with `#chatClientSelect`).
- `#addClientBtn`: Button to open "New Client" modal.
- `#uploadZone`: Drag & drop area for files.
- `#fileInput`: Hidden file input element.
- `#uploadOptions`: Container for upload settings.
- `#useOcrCheckbox`: Checkbox for OCR setting.
- `#fastModeCheckbox`: Checkbox for Fast Mode setting.
- `#uploadProgress`: Container for progress bar.
- `#progressBar`: The progress bar element.
- `#progressText`: Text displaying upload status.
- `#documentList`: Container for uploaded document items.
- `.delete-doc-btn`: Class for delete button within document item.

### Modals & Overlays
- `#toastContainer`: Container for toast notifications.
- `#loadingOverlay`: Full-screen loading overlay.
- `#loadingText`: Text inside loading overlay.
- `#clientModal`: Modal for creating a new client.
- `#closeClientModal`: Button to close client modal.
- `#cancelClientBtn`: Button to cancel client creation.
- `#createClientBtn`: Button to submit new client.
- `#clientName`: Input for new client name.

## 2. `frontend/login.html` (Authentication)

### Login Form
- `#loginForm`: The form element.
- `#username`: Input for username.
- `#password`: Input for password.
- `#togglePassword`: Button to toggle password visibility.
- `#submitBtn`: Submit button (has loading state).
- `#errorMessage`: Container for error alerts.
- `#errorText`: Text inside error message.

## 3. `frontend/profile.html` (Profile & Admin)

### Layout & Navigation
- `#app`: Main application container (hidden until auth check).
- `#tabs`: Container for navigation tabs.
- `#logoutBtn`: Button to logout.
- `#userBadge`: Header element displaying user role.
- `#content`: Main content area where tabs are rendered.

### Profile Tab
- `#currentPass`: Input for current password.
- `#newPass`: Input for new password.
- `#confirmPass`: Input for confirming new password.
- `#changePassBtn`: Button to update password.
- `#passMsg`: Feedback text element.

### Audit Tab (Admin)
- `#filterUser`: Input for filtering by username.
- `#filterAction`: Dropdown for filtering by action.
- `#filterMethod`: Dropdown for filtering by HTTP method.
- `#filterStatus`: Input for filtering by status code.
- `#applyFilters`: Button to apply filters.
- `#auditPanel`: Container for audit log table.

### Users Tab (Admin)
- `#userList`: Container for user table.
- `#newUser`: Input for new username.
- `#newEmail`: Input for new email.
- `#newPass`: Input for new password.
- `#newRole`: Dropdown for new user role.
- `#createUserBtn`: Button to create user.
- `#userMsg`: Feedback text element.
- `#editUserPanel`: Panel/Modal for editing user.
- `#editUserId`: Hidden input for user ID.
- `#editEmail`: Input for editing email.
- `#editPass`: Input for editing password.
- `#editRole`: Dropdown for editing role.
- `#editStatus`: Dropdown for editing status.
- `#saveUserBtn`: Button to save user changes.
- `#cancelEditBtn`: Button to cancel editing.
- `#editMsg`: Feedback text element.

### Clients Tab
- `#clientList`: Container for clients table.
- `#newClientName`: Input for new client name.
- `#newClientAliases`: Input for new client aliases.
- `#createClientBtn`: Button to create client.
- `#clientMsg`: Feedback text element.
- `#userAccessPanel`: Container for access management.
- `#accessClientSelect`: Dropdown for selecting client.
- `#accessUserSelect`: Dropdown for selecting user.
- `#assignUserBtn`: Button to grant access.
- `#revokeUserBtn`: Button to revoke access.
- `#accessMsg`: Feedback text element.
- `#clientUsersPreview`: Container for showing assigned users.

### Evaluation Tab (Admin)
- `#datasetPanel`: Container for datasets table.
- `#runPanel`: Container for runs table.
- `#dsName`: Input for dataset name.
- `#dsClient`: Input for dataset client ID.
- `#dsSize`: Input for sample size.
- `#genBtn`: Button to generate dataset.
- `#genMsg`: Feedback text element.
- `#runDataset`: Input for dataset ID.
- `#runK`: Input for 'k' value.
- `#runBtn`: Button to run evaluation.
- `#runMsg`: Feedback text element.

### Stats Tab (Admin)
- `#statPanel`: Container for key stats.
- `#configPanel`: Container for configuration table.
