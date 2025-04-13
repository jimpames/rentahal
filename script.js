// TERMS AND CONDITIONS

// ## 🔐 Supplemental License Terms (RENT A HAL Specific)

// In addition to the terms of the GNU General Public License v3.0 (GPL-3.0), the following conditions **explicitly apply** to this project and all derivative works:

// - 🚫 **No Closed Source Derivatives**: Any derivative, fork, or modified version of RENT A HAL must **remain fully open source** under a GPL-compatible license.
  
// - 🧬 **No Patents**: You **may not patent** RENT A HAL or any part of its original or derived code, design, architecture, or functional implementations.

// - 🔁 **License Must Propagate**: Any distribution of modified versions must include this exact clause, in addition to the GPL-3.0, to ensure **eternal openness**.

// - ⚖️ **Enforcement**: Violation of these conditions terminates your rights under this license and may be pursued legally.

// This clause is intended to **protect the freedom and integrity** of this AI system for all present and future generations. If you use it — respect it.

// > "This project is free forever. If you change it — it stays free too."

// this notice must remain in all copies / derivatives of the work forever and must not be removed.

document.addEventListener('DOMContentLoaded', function() {
    // WebSocket connection
	checkForOAuthCallback();
    let socket;
    let reconnectInterval = 1000; // Start with 1 second interval
    let reconnectTimer;
    let heartbeatInterval;
    const MAX_RECONNECT_INTERVAL = 30000; // Maximum reconnect interval: 30 seconds
	let gmailCommandAttempts = 0;
	const MAX_GMAIL_COMMAND_ATTEMPTS = 3;


    // Wake word variables
    let wakeWordState = 'inactive'; // 'inactive', 'listening', 'menu', 'prompt', 'processing'
    let wakeWordRecognition;
    let currentPrompt = '';
    let isListening = false;
    let isSystemSpeaking = false;
    let isWakeWordModeActive = false; // Default to false or adjust as needed
	let isRestarting = false; // Flag to prevent overlapping restarts
    let inactivityCount = 0;
	let inactivityTimer;
	let promptInactivityCount = 0;
	let promptInactivityTimer;


    // Audio visualization variables
    let audioContext;
    let analyser;
    let dataArray;
    let canvasCtx;
    let animationId;

	let audioQueue = [];
	let isAudioPlaying = false;
	let isTTSPlaying = false;
	let speechSynthesis = window.speechSynthesis;
	let queryAlreadyRun = false; // Track if a query has been run


	let authHandled = false;



	// gmail client ID = somenumbers-aguidlookingthingy.apps.googleusercontent.com
	// test user = rentahal9000@gmail.com		
	// client secret = hadyouforaminutedidnti_-guidthingy
	// api key = bignumberssecretthingy
	
	
	
	
	// need to put these two lines below in the index.html at the bottom just before the closing slash body tag /body 
	
	// <script async defer src="https://apis.google.com/js/api.js" onload="gapiLoaded()"></script>
	// <script async defer src="https://accounts.google.com/gsi/client" onload="gisLoaded()"></script>
	
	
	

	// Gmail API variables
	const CLIENT_ID = 'nonnoicanttellyou.apps.googleusercontent.com';
	const API_KEY = 'thisisapretendkey';
	const DISCOVERY_DOC = 'https://www.googleapis.com/discovery/v1/apis/gmail/v1/rest';
	const SCOPES = 'https://www.googleapis.com/auth/gmail.readonly';
	let tokenClient;
	let gapiInited = false;
	let gisInited = false;




	// Check for OAuth callback
	if (window.location.hash.includes('access_token')) {
		const params = new URLSearchParams(window.location.hash.substring(1));
		const accessToken = params.get('access_token');
		const state = params.get('state');
		handleOAuthCallback(accessToken, state);
	}










	// Add this function at the beginning of your script
	function checkForOAuthCallback() {
		const hash = window.location.hash.substring(1);
		const params = new URLSearchParams(hash);
		const accessToken = params.get('access_token');
		const state = params.get('state');

		if (accessToken && state) {
			handleOAuthCallback(accessToken, state);
			// Clear the hash to remove the token from the URL
			history.replaceState(null, null, ' ');
		}
	}


















	function gapiLoaded() {
		// Initialize the Google Identity Services (GIS) client
		tokenClient = google.accounts.oauth2.initTokenClient({
			client_id: CLIENT_ID,
			scope: SCOPES,
			callback: (resp) => {
				if (resp.error !== undefined) {
					console.error("Gmail auth error:", resp.error);
				} else {
					console.log("Gmail auth successful");
					// Save the token and trigger Gmail API access here
					localStorage.setItem('gmail_access_token', resp.access_token);
					loadGmailApi();  // Call to load Gmail API after successful auth
				}
			}
		});
		gapiInited = true;

		// Load Gmail API directly if access token is already present
		if (localStorage.getItem('gmail_access_token')) {
			loadGmailApi();  // Load Gmail API if token is found
		}
	}






	async function initializeGapiClient() {
		await gapi.client.init({
			apiKey: API_KEY,
			discoveryDocs: [DISCOVERY_DOC],
		});
		gapiInited = true;
		checkAuthAndReadEmails(); // Directly call the readback here
	}

	function gisLoaded() {
		tokenClient = google.accounts.oauth2.initTokenClient({
			client_id: CLIENT_ID,
			scope: SCOPES,
			callback: (resp) => {
				if (resp.error !== undefined) {
					console.error("Gmail auth error:", resp.error);
				} else {
					console.log("Gmail auth successful");
					localStorage.setItem('gmail_access_token', resp.access_token); // Save the token
					checkAuthAndReadEmails(); // Trigger reading emails
				}
			}
		});
		gisInited = true;
	}

	// Function to check authentication and read emails
	function checkAuthAndReadEmails() {
		const token = localStorage.getItem('gmail_access_token');
		if (token) {
			gapi.client.gmail.users.messages.list({
				'userId': 'me',
				'labelIds': 'INBOX',
				'maxResults': 10
			}).then(response => {
				const messages = response.result.messages;
				console.log('Email messages:', messages);
				if (messages && messages.length > 0) {
					messages.forEach(message => {
						getEmailDetails(message.id);
					});
				}
			}).catch(error => {
				console.error("Error loading emails:", error);
			});
		} else {
			console.log("No valid Gmail access token found.");
		}
	}
























	function initiateGmailAuth() {
		const accessToken = localStorage.getItem('gmail_access_token');
		if (!accessToken) {
			console.log("No access token found in local storage, opening Gmail authorization window.");

			const clientId = 'nownownowwecanttellthis.apps.googleusercontent.com';
			const redirectUri = encodeURIComponent('https://rentahal.com/static/oauth-callback.html');
			const scope = encodeURIComponent('https://www.googleapis.com/auth/gmail.readonly');
			const state = encodeURIComponent(generateRandomState());

			const authUrl = `https://accounts.google.com/o/oauth2/v2/auth?` +
				`client_id=${clientId}&` +
				`redirect_uri=${redirectUri}&` +
				`response_type=token&` +
				`scope=${scope}&` +
				`state=${state}&` +
				`include_granted_scopes=true`;

			const authWindow = window.open(authUrl, 'Gmail Authorization', 'width=600,height=600');
        
			// Log for message tracking
			console.log("Gmail authorization window opened.");

			// Message event listener
			window.addEventListener('message', function(event) {
				console.log("Received message event:", event);

				// Ensure the event is coming from the expected origin
				if (event.origin !== "https://rentahal.com") {
					console.warn("Event origin does not match, ignoring message.");
					return;
				}

				// Handle OAuth callback
				if (event.data.type === 'OAUTH_CALLBACK') {
					console.log("Received OAUTH_CALLBACK message.");
					if (event.data.accessToken) {
						console.log("Access token found, saving to local storage.");
						localStorage.setItem('gmail_access_token', event.data.accessToken);
						handleOAuthCallback(event.data.accessToken, event.data.state);
					} else {
						console.error("Error: No access token found in callback data.");
					}
				}

				// Handle closing the window
				if (event.data.type === 'OAUTH_CLOSE_WINDOW') {
					console.log("Closing OAuth window.");
					if (authWindow) authWindow.close();  // Ensure the window is closed properly
				}
			}, false);
		} else {
			console.log("Access token found in local storage, skipping authentication.");
			handleGmailAuthSuccess();  // Proceed to reading emails
		}
	}








	function generateRandomState() {
		return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
	}






	function handleOAuthCallback(event) {
		const oauthResponse = event.data;  // Assuming the token comes in this event
		console.log("OAuth response received:", oauthResponse);

		if (oauthResponse && oauthResponse.access_token) {
			// Log access token and store it in local storage
			console.log("Access token found, storing:", oauthResponse.access_token);
			localStorage.setItem('gmail_access_token', oauthResponse.access_token);

			// Close the OAuth window
			try {
				window.close();
			} catch (e) {
				console.error("Failed to close window:", e);
			}

		} else if (oauthResponse && !oauthResponse.access_token) {
			// Handle missing access token
			console.error("OAuth response received but no access token found. Response data:", oauthResponse);
		} else {
			// Log if the oauthResponse itself is null or undefined
			console.error("Failed to retrieve access token or response is invalid.");
		}
	}







	async function handleGmailAuthSuccess() {
		if (wakeWordState !== 'gmail') {
			wakeWordState = 'gmail';
			try {
				await loadGmailApi();
				console.log("Gmail API loaded successfully");
				speakFeedback("Gmail ready. Starting to read your emails.", () => {
					startReadingEmails(); // Immediate read
				});
			} catch (error) {
				console.error("Error loading Gmail API:", error);
				speakFeedback(`Error initializing Gmail: ${error.message || error}. Please try again later.`, () => {
					wakeWordState = 'listening';
					handleTopLevelCommand("computer");
				});
			}
		} else {
			console.log("Gmail mode already active");
		}
	}








	function handleGmailAuthFailure() {
		speakFeedback("I couldn't access your Gmail account. Please try again later.", () => {
			wakeWordState = 'listening';
			handleTopLevelCommand("computer");
		});
	}	
	


	async function startReadingEmails() {
		try {
			const emails = await readEmails();
			if (emails && emails.length > 0) {
				await readEmailsOneByOne(emails);
			} else {
				speakFeedback("No new emails found.", () => {
					wakeWordState = 'listening';
					handleTopLevelCommand("computer");
				});
			}
		} catch (error) {
			console.error("Error reading emails:", error);
			speakFeedback("An error occurred while reading your emails. Please try again later.", () => {
				wakeWordState = 'listening';
				handleTopLevelCommand("computer");
			});
		}
	}



	async function readEmailsOneByOne(emails) {
		let currentIndex = 0;

		while (currentIndex < emails.length) {
			const email = emails[currentIndex];
			console.log(`Reading email ${currentIndex + 1} of ${emails.length}`);

			await new Promise(resolve => {
				const emailContent = `Email ${currentIndex + 1} of ${emails.length}. From ${email.from}: Subject: ${email.subject}`;
				speakFeedback(emailContent, resolve);
			});

			let awaitingCommand = true;
			while (awaitingCommand) {
				await new Promise(resolve => {
					speakFeedback("Say 'next' for the next email or 'finish' to stop.", resolve);
				});

				const command = await waitForNextCommandWithTimeout(20000);
				console.log(`Received command: ${command}`);

				if (command === "timeout") {
					await new Promise(resolve => {
						speakFeedback("No command received. Please try again.", resolve);
					});
				} else if (command && command.includes("finish")) {
					awaitingCommand = false;
					currentIndex = emails.length; // Exit the outer loop
				} else if (command && command.includes("next")) {
					currentIndex++;
					awaitingCommand = false;
				} else {
					await new Promise(resolve => {
						speakFeedback("Command not recognized. Please say 'next' or 'finish'.", resolve);
					});
				}
			}
		}

		speakFeedback("Email reading finished. Returning to main menu.", () => {
			wakeWordState = 'listening';
			handleTopLevelCommand("computer");
		});
	}





	function waitForNextCommandWithTimeout(timeout) {
		return new Promise((resolve) => {
			if (wakeWordRecognition.state === 'listening') {
				wakeWordRecognition.stop();
			}

			const timer = setTimeout(() => {
				resolve("timeout");
			}, timeout);

			wakeWordRecognition.onresult = function(event) {
				clearTimeout(timer);
				const last = event.results.length - 1;
				const command = event.results[last][0].transcript.trim().toLowerCase();
				resolve(command);
			};

			wakeWordRecognition.onerror = function(event) {
				clearTimeout(timer);
				console.error("Speech recognition error:", event.error);
				resolve("error");
			};

			wakeWordRecognition.onend = function() {
				// Do nothing; we'll restart if needed
			};

			try {
				wakeWordRecognition.start();
			} catch (error) {
				console.error("Error starting speech recognition:", error);
				resolve("error");
			}
		});
	}





	async function readEmails() {
		console.log("Attempting to read emails");

		const accessToken = localStorage.getItem('gmail_access_token');
		if (!accessToken) {
			console.error("No access token found. Initiating Gmail authentication.");
			initiateGmailAuth();
			return;
		}

		try {
			if (!gapi.client.gmail) {
				await gapi.client.load('gmail', 'v1');
			}

			gapi.auth.setToken({ access_token: accessToken });

			const response = await gapi.client.gmail.users.messages.list({
				'userId': 'me',
				'maxResults': 20
			});

			const messages = response.result.messages;
			if (!messages || messages.length === 0) {
				console.log("No emails found");
				await speakFeedback("No new emails found.");
				return;
			}

			console.log("Emails found:", messages.length);
        
			let currentIndex = 0;
			const batchSize = 5;

			while (currentIndex < messages.length) {
				const endIndex = Math.min(currentIndex + batchSize, messages.length);
				await speakFeedback(`Reading emails ${currentIndex + 1} to ${endIndex}`);

				for (let i = currentIndex; i < endIndex; i++) {
					const emailDetails = await getEmailDetails(messages[i].id);
					await speakFeedback(`Email ${i + 1}: From ${emailDetails.from}. Subject: ${emailDetails.subject}`);
                
					if (i < endIndex - 1) {
						const command = await waitForNextCommandWithTimeout(10000);
						if (command.includes("finish")) {
							return;
						} else if (!command.includes("next")) {
							await speakFeedback("Command not recognized. Moving to next email.");
						}
					}
				}

				currentIndex = endIndex;

				if (currentIndex < messages.length) {
					await speakFeedback("End of batch. Say 'next' for the next batch or 'finish' to stop.");
					const command = await waitForNextCommandWithTimeout(10000);
					if (command.includes("finish")) {
						break;
					} else if (!command.includes("next")) {
						await speakFeedback("Command not recognized. Moving to next batch.");
					}
				}
			}

			await speakFeedback("All emails have been read. Returning to main menu.");
		} catch (err) {
			console.error('Error reading emails:', err);
			await speakFeedback("An error occurred while reading emails. Returning to main menu.");
		} finally {
			wakeWordState = 'listening';
			handleTopLevelCommand("computer");
		}
	}





	async function getEmailDetails(messageId) {
		try {
			const response = await gapi.client.gmail.users.messages.get({
				'userId': 'me',
				'id': messageId
			});
			const message = response.result;
			const headers = message.payload.headers;
			const subject = headers.find(header => header.name === "Subject")?.value || "No subject";
			const from = headers.find(header => header.name === "From")?.value || "Unknown sender";
			return { subject, from };
		} catch (err) {
			console.error('Error getting email details:', err);
			return { subject: 'Error retrieving subject', from: 'Error retrieving sender' };
		}
	}



	function waitForNextCommandWithTimeout(timeout) {
		return new Promise((resolve) => {
			const timer = setTimeout(() => {
				resolve("timeout");
			}, timeout);

			function onResult(event) {
				clearTimeout(timer);
				const last = event.results.length - 1;
				const command = event.results[last][0].transcript.trim().toLowerCase();
				wakeWordRecognition.removeEventListener('result', onResult);
				resolve(command);
			}

			wakeWordRecognition.addEventListener('result', onResult);

			if (wakeWordRecognition.state !== 'listening') {
				wakeWordRecognition.start();
			}
		});
	}













	
	
	// Gmail API functions

	// async function listLabels() {
	// 	console.log("Attempting to list labels");
	// 	if (!gapi.client.gmail) {
	// 		console.error("Gmail API not loaded");
	// 		await loadGmailApi();
	// 	}
	// 	try {
	// 		const response = await gapi.client.gmail.users.labels.list({
	// 			'userId': 'me',
	// 		});
	// 		const labels = response.result.labels;
	// 		if (!labels || labels.length === 0) {
	// 			console.log("No labels found");
	// 			speakFeedback("No labels found in your Gmail account.", startGmailCommandLoop);
	// 			return [];
	// 		} else {
	// 			console.log("Labels found:", labels);
	// 			const labelNames = labels.map(label => label.name).join(", ");
	// 			speakFeedback(`Your Gmail labels are: ${labelNames}`, startGmailCommandLoop);
	// 			return labels;
	// 		}
	// 	} catch (err) {
	// 		console.error('Error listing labels:', err);
	// 		speakFeedback("Sorry, I couldn't retrieve your Gmail labels. Please try again.", startGmailCommandLoop);
	// 		return [];
	// 	}
	// }



	async function loadGmailApi() {
		return new Promise((resolve, reject) => {
			if (!gapiInited) {
				gapi.load('client', async () => {
					try {
						await gapi.client.init({
							apiKey: API_KEY,
							discoveryDocs: [DISCOVERY_DOC],
						});
						console.log("Gmail API initialized and loaded");
						resolve();
					} catch (error) {
						console.error("Error initializing Gmail API:", error);
						reject(error);
					}
				});
			} else {
				gapi.client.load('gmail', 'v1', () => {
					console.log("Gmail API loaded");
					resolve();
				});
			}
		});
	}





	async function handleGmailCommands(command) {
		console.log("Processing Gmail command:", command);

		if (command.includes("read") || command.includes("mail")) {
			console.log("Read email command recognized");
			try {
				const emailDetails = await readEmails();
				if (emailDetails && emailDetails.length > 0) {
					const emailMessage = `You have ${emailDetails.length} unread emails. The first email is from ${emailDetails[0].from}, with the subject: ${emailDetails[0].subject}.`;
					speakFeedback(emailMessage, startGmailCommandLoop);
				} else {
					speakFeedback("No unread emails found.", startGmailCommandLoop);
				}
			} catch (error) {
				console.error("Error reading emails:", error);
				speakFeedback("An error occurred while reading emails. Please try again.", startGmailCommandLoop);
			}
		} else if (command.includes("sign out") || command.includes("signout")) {
			console.log("Sign out command recognized");
			handleGmailSignout();
			speakFeedback("Signed out of Gmail.", () => {
				wakeWordState = 'listening';
				handleTopLevelCommand("computer");
			});
		} else {
			console.error("Unrecognized Gmail command:", command);
			speakFeedback("Unrecognized mail command. Please try again.", startGmailCommandLoop);
		}
	}



	function handleGmailSignout() {
		localStorage.removeItem('gmail_access_token');
		// Add any additional sign-out logic here
		console.log("User signed out of Gmail");
	}
















	const persistentAudio = new Audio();

    // DOM elements
    const userInfo = document.getElementById('user-info');
    const nicknameInput = document.getElementById('nickname-input');
    const setNicknameButton = document.getElementById('set-nickname');
    const promptInput = document.getElementById('prompt-input');
    const queryType = document.getElementById('query-type');
    const modelType = document.getElementById('model-type');
    const modelSelect = document.getElementById('model-select');
    const imageUpload = document.getElementById('image-upload');
    const imagePreview = document.getElementById('image-preview');
    const previewImg = document.getElementById('preview-img');
    const submitQueryButton = document.getElementById('submit-query');
    const voiceInputButton = document.getElementById('voice-input-button');
    const speechOutputCheckbox = document.getElementById('speech-output-checkbox');
    const results = document.getElementById('results');
    const queueThermometer = document.getElementById('queue-thermometer');
    const previousQueries = document.getElementById('previous-queries');
    const sysopPanel = document.getElementById('sysop-panel');
    const workerList = document.getElementById('worker-list');
    const huggingFaceModelList = document.getElementById('huggingface-model-list');
    const userList = document.getElementById('user-list');
    const sysopMessageInput = document.getElementById('sysop-message-input');
    const sendSysopMessageButton = document.getElementById('send-sysop-message');
    const systemStats = document.getElementById('system-stats');
    const cumulativeCosts = document.getElementById('cumulative-costs');
    const connectionStatus = document.getElementById('connection-status');
    const clearResultsButton = document.getElementById('clear-results');
    const activeUsersTable = document.getElementById('active-users-table').getElementsByTagName('tbody')[0];
    const toggleWakeWordButton = document.getElementById('toggle-wake-word');
    const audioWaveform = document.getElementById('audioWaveform');

    let currentUser = null;
    let huggingFaceModels = {};
    let aiWorkers = {};

    // Chunking constants
    const CHUNK_SIZE = 1024 * 1024; // 1MB chunks

    // Voice recording variables
    let isRecording = false;
    let mediaRecorder;
    let audioChunks = [];
    let speechOutputEnabled = false;



















// Add this function to your script
	function setupAudioHandling() {
		persistentAudio.addEventListener('ended', playNextAudio);
		persistentAudio.addEventListener('error', handleAudioError);
		document.body.appendChild(persistentAudio);
	}













    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        socket = new WebSocket(wsUrl);

        socket.onopen = (event) => {
            console.log('WebSocket connection opened:', event);
            displayStatus('Connected to server');
            updateConnectionStatus(true);
            clearTimeout(reconnectTimer);
            reconnectInterval = 1000; // Reset reconnect interval on successful connection
            startHeartbeat();
            sendToWebSocket({ type: 'get_previous_queries' });
        };

        socket.onmessage = (event) => {
            const message = JSON.parse(event.data);
            console.log('Received message:', message);

            switch (message.type) {
                case 'set_cookie':
                    document.cookie = `${message.name}=${message.value}; path=/; max-age=31536000; SameSite=Strict`;
                    break;
                case 'user_info':
                    handleUserInfo(message.data);
                    init(); // Call init after receiving user info
                    break;
                case 'previous_queries':
                    displayPreviousQueries(message.data);
                    break;
                case 'query_result':
                    handleQueryResult(message.result, message.processing_time, message.cost, message.result_type);
                    break;
                case 'queue_update':
                    updateQueueStatus(message.depth, message.total);
                    break;
                case 'system_stats':
                    updateSystemStats(message.data);
                    break;
                case 'user_stats':
                    updateUserStats(message.data);
                    break;
                case 'worker_health':
                    updateWorkerHealth(message.data);
                    break;
                case 'worker_update':
                    updateWorkerList(message.workers);
                    break;
                case 'huggingface_update':
                    updateHuggingFaceModelList(message.models);
                    break;
                case 'user_banned':
                    handleUserBanned(message.guid);
                    break;
                case 'user_unbanned':
                    handleUserUnbanned(message.guid);
                    break;
                case 'query_terminated':
                    handleQueryTerminated(message.guid);
                    break;
                case 'sysop_message':
                    displaySysopMessage(message.message);
                    break;
                case 'active_users':
                    updateActiveUsers(message.users);
                    break;
                case 'error':
                    displayError(message.message);
                    break;
                case 'ping':
                    sendToWebSocket({ type: 'pong' });
                    break;
                case 'vision_upload_complete':
                    displayStatus(message.message);
                    break;
                case 'vision_chunk_received':
                    // Optionally update UI to show progress
                    break;
                case 'transcription_result':
                    handleTranscriptionResult(message.text);
                    break;
                case 'speech_result':
                    handleSpeechResult(message.audio);
                    break;
            }
        };

        socket.onclose = (event) => {
            console.log('WebSocket connection closed:', event);
            displayError("Connection lost. Attempting to reconnect...");
            updateConnectionStatus(false);
            clearInterval(heartbeatInterval);
            scheduleReconnection();
        };

        socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            displayError("WebSocket error occurred. Please check your connection.");
            updateConnectionStatus(false);
        };
    }

    function scheduleReconnection() {
        clearTimeout(reconnectTimer);
        reconnectTimer = setTimeout(() => {
            connectWebSocket();
        }, reconnectInterval);
        reconnectInterval = Math.min(reconnectInterval * 2, MAX_RECONNECT_INTERVAL);
    }

    function startHeartbeat() {
        clearInterval(heartbeatInterval);
        heartbeatInterval = setInterval(() => {
            if (socket.readyState === WebSocket.OPEN) {
                sendToWebSocket({ type: 'pong' });
            }
        }, 25000); // Send heartbeat every 25 seconds
    }

    // Event listeners
    if (setNicknameButton) setNicknameButton.addEventListener('click', setNickname);
    if (submitQueryButton) submitQueryButton.addEventListener('click', handleSubmitQuery);
    if (queryType) queryType.addEventListener('change', handleQueryTypeChange);
    if (modelType) modelType.addEventListener('change', handleModelTypeChange);
    if (imageUpload) imageUpload.addEventListener('change', handleImageUpload);
    if (sendSysopMessageButton) sendSysopMessageButton.addEventListener('click', sendSysopMessage);
    if (clearResultsButton) {
        clearResultsButton.addEventListener('click', clearResults);
        clearResultsButton.title = "Clear displayed results (does not affect database)";
    }
    if (voiceInputButton) voiceInputButton.addEventListener('click', toggleVoiceRecording);
    if (speechOutputCheckbox) speechOutputCheckbox.addEventListener('change', toggleSpeechOutput);
    if (toggleWakeWordButton) toggleWakeWordButton.addEventListener('click', toggleWakeWordMode);

    // Function implementations
    function handleUserInfo(user) {
        currentUser = user;
        if (userInfo) userInfo.textContent = `User: ${user.nickname} (${user.guid})`;
        if (sysopPanel) {
            sysopPanel.style.display = user.is_sysop ? 'block' : 'none';
            if (user.is_sysop) {
                // Immediately request stats when identified as sysop
                sendToWebSocket({ type: 'get_stats' });
            }
        }
        updateCumulativeCosts(user);
    }

    function setNickname() {
        const newNickname = nicknameInput.value.trim();
        if (newNickname) {
            sendToWebSocket({
                type: 'set_nickname',
                nickname: newNickname
            });
        }
    }

    function setupAudioVisualization() {
        console.log("Setting up audio visualization");
        canvasCtx = audioWaveform.getContext('2d');

        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        const bufferLength = analyser.frequencyBinCount;
        dataArray = new Uint8Array(bufferLength);

        // Connect the microphone to the analyser
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                const source = audioContext.createMediaStreamSource(stream);
                source.connect(analyser);
                drawWaveform();
            })
            .catch(err => {
                console.error('Error accessing microphone:', err);
                displayError('Error accessing microphone. Please ensure you have given permission.');
            });
    }

	function drawWaveform() {
		if (!analyser) {
			console.log("Analyser not initialized");
			return;
		}
		animationId = requestAnimationFrame(drawWaveform);
	
		analyser.getByteTimeDomainData(dataArray);

		canvasCtx.fillStyle = 'rgb(200, 200, 200)';
		canvasCtx.fillRect(0, 0, audioWaveform.width, audioWaveform.height);

		canvasCtx.lineWidth = 2;
		canvasCtx.strokeStyle = 'rgb(0, 0, 0)';

		canvasCtx.beginPath();

		const sliceWidth = audioWaveform.width * 1.0 / analyser.frequencyBinCount;
		let x = 0;

		for (let i = 0; i < analyser.frequencyBinCount; i++) {
			const v = dataArray[i] / 128.0;
			const y = v * audioWaveform.height / 2;

			if (i === 0) {
				canvasCtx.moveTo(x, y);
			} else {
				canvasCtx.lineTo(x, y);
			}

			x += sliceWidth;
		}

		canvasCtx.lineTo(audioWaveform.width, audioWaveform.height / 2);
		canvasCtx.stroke();
	}

    function initializeWakeWordRecognition() {
        console.log("Initializing wake word recognition");
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SpeechRecognition) {
            console.log("SpeechRecognition is supported");
            wakeWordRecognition = new SpeechRecognition();
            wakeWordRecognition.lang = 'en-US';
            wakeWordRecognition.interimResults = false;
            wakeWordRecognition.maxAlternatives = 1;
            wakeWordRecognition.continuous = false;

            wakeWordRecognition.onstart = function() {
                console.log("Wake word recognition started");
                isListening = true;
                try {
                    if (typeof setupAudioVisualization === 'function') {
                        setupAudioVisualization();
                        if (audioWaveform) {
                            audioWaveform.style.display = 'block';
                        }
                    }
                } catch (error) {
                    console.error("Error in audio visualization setup:", error);
                }
            };

            wakeWordRecognition.onend = function() {
                console.log("Wake word recognition ended");
                isListening = false; // Reset the state
                isRestarting = false; // Reset the restart flag

                if (wakeWordState !== 'inactive') {
                    console.log("Restarting wake word recognition");
                    setTimeout(() => {
                        if (wakeWordState !== 'inactive' && !isListening) {
                            try {
                                wakeWordRecognition.start();
                                isListening = true;
                            } catch (error) {
                                console.error("Error restarting recognition:", error);
                            }
                        }
                    }, 1000); // Add a delay to prevent rapid restart attempts
                } else {
                    if (audioWaveform) {
                        audioWaveform.style.display = 'none';
                    }
                    if (typeof animationId !== 'undefined' && animationId) {
                        cancelAnimationFrame(animationId);
                    }
                }
            };
            wakeWordRecognition.onresult = function(event) {
                console.log("Speech recognition result received");
                const last = event.results.length - 1;
                const command = event.results[last][0].transcript.trim().toLowerCase();
                console.log("Heard: ", command);

                if (isSystemSpeaking) {
                    console.log("Ignoring speech recognition result as system is speaking");
                    return;
                }

                switch (wakeWordState) {
                    case 'listening':
                        handleTopLevelCommand(command);
                        break;
                    case 'menu':
                        handleMenuCommand(command);
                        break;
                    case 'prompt':
                        handlePromptInput(command);
                        break;
                }
            };






            wakeWordRecognition.onerror = function(event) {
                console.error('Wake word recognition error:', event.error);
                displayError(`Wake word recognition error: ${event.error}`);
                isListening = false; // Reset the state in case of an error

                if (isRestarting) {
                    console.warn("Recognition restart already in progress. Skipping restart.");
                    return;
                }

                if (wakeWordState !== 'inactive') {
                    console.log("Restarting wake word recognition after error");
                    isRestarting = true;
                    setTimeout(() => {
                        if (wakeWordState !== 'inactive' && !isListening) {
                            try {
                                wakeWordRecognition.start();
                                isListening = true;
                            } catch (error) {
                                console.error("Error restarting recognition:", error);
                            } finally {
                                isRestarting = false; // Reset the flag
                            }
                        }
                    }, 1000); // Adjust the delay as necessary
                }
            };





            // Start recognition immediately
            wakeWordRecognition.start();
        } else {
            console.error("Speech recognition not supported in this browser");
            displayError("Wake word mode is not supported in your browser. Please try using Chrome or Edge.");
        }
    }








	async function callWebcamVisionRoutine() {
		speakFeedback("Accessing webcam for vision processing.", async () => {
			const video = await setupCamera();
			if (video) {
				showStaticWaveform();
				await new Promise(resolve => setTimeout(resolve, 1000)); // Give the camera a moment to adjust
				const imageData = await captureImage(video);
				stopCamera();
				await processVisionQuery(imageData);
			} else {
				wakeWordState = 'listening';
				handleTopLevelCommand("computer");
			}
		});
	}







	async function setupCamera() {
		try {
			const stream = await navigator.mediaDevices.getUserMedia({ video: true });
			const video = document.createElement('video');
			video.srcObject = stream;
			video.style.display = 'none'; // Hide the video element
			document.body.appendChild(video); // Add to DOM
			await video.play();
			return video;
		} catch (error) {
			console.error('Error accessing webcam:', error);
			displayError('Error accessing webcam. Please ensure you have given permission to use the camera.');
			return null;
		}
	}








	async function captureImage(video) {
		const canvas = document.createElement('canvas');
		canvas.width = video.videoWidth;
		canvas.height = video.videoHeight;
		canvas.getContext('2d').drawImage(video, 0, 0);
		return canvas.toDataURL('image/jpeg');
	}

	function stopCamera() {
		const video = document.querySelector('video');
		if (video && video.srcObject) {
			const stream = video.srcObject;
			const tracks = stream.getTracks();
			tracks.forEach(track => track.stop());
			video.srcObject = null;
		}
	}

	async function processVisionQuery(imageData) {
		const query = {
			type: 'submit_query',
			query: {
				prompt: "Describe this image in detail",
				query_type: "vision",
				model_type: "worker_node",
				model_name: "default_vision_model",
				image: imageData.split(',')[1] // Remove the "data:image/jpeg;base64," part
			}
		};

		sendToWebSocket(query);
	}



	function displayCapturedImage(imageData) {
		const imageContainer = document.createElement('div');
		imageContainer.id = 'captured-image-container';
		imageContainer.style.position = 'fixed';
		imageContainer.style.top = '20px';
		imageContainer.style.right = '20px';
		imageContainer.style.zIndex = '1000';

		const image = document.createElement('img');
		image.src = imageData;
		image.style.maxWidth = '300px';
		image.style.border = '2px solid #333';
		image.style.borderRadius = '10px';

		imageContainer.appendChild(image);
		document.body.appendChild(imageContainer);

		// Remove the image after 10 seconds
		setTimeout(() => {
			document.body.removeChild(imageContainer);
		}, 10000);
	}






























	function speakFeedback(message, callback) {
		console.log("Starting TTS playback.");

		const utterance = new SpeechSynthesisUtterance(message);

		utterance.onend = () => {
			console.log("Finished speaking feedback.");

			if (callback) callback();
        
			// Re-enable wake word mode
			setTimeout(() => {
				console.log("Enabling speech recognition for wake word mode.");
				enableSpeechRecognition(callback);
			}, 500); // 500 ms delay
		};

		utterance.onerror = (event) => {
			console.error("Error in speech synthesis:", event.error);

			if (callback) callback();

			// Re-enable wake word mode even if there was an error
			setTimeout(() => {
				console.log("Enabling speech recognition for wake word mode.");
				enableSpeechRecognition(callback);
			}, 500); // 500 ms delay
		};

		try {
			window.speechSynthesis.speak(utterance);
		} catch (error) {
			console.error("Speech synthesis error:", error);

			if (callback) callback();

			// Re-enable wake word mode after error
			setTimeout(() => {
				console.log("Enabling speech recognition for wake word mode.");
				enableSpeechRecognition(callback);
			}, 500); // 500 ms delay
		}
	}











    function toggleWakeWordMode() {
        console.log("Toggling wake word mode");
        if (wakeWordState === 'inactive') {
            activateWakeWordMode();
        } else {
            deactivateWakeWordMode();
        }
    }




    function activateWakeWordMode() {
        if (wakeWordState === 'listening' || isListening) {
            console.warn("Wake word mode is already active.");
            return; // Prevent re-activation if already active
        }

        console.log("Activating wake word mode");
        wakeWordState = 'listening';
        isWakeWordModeActive = true; // Ensure this is set to true
        toggleWakeWordButton.textContent = "Disable Wake Word Mode";
        toggleWakeWordButton.classList.remove('bg-blue-500');
        toggleWakeWordButton.classList.add('bg-red-500');

        initializeWakeWordRecognition();
        showWaveform();

        speakAndListen("9000 ready. Say the wake word.", handleTopLevelCommand);
    }






	function deactivateWakeWordMode() {
		console.log("Deactivating wake word mode");
		wakeWordState = 'inactive';
		isWakeWordModeActive = false;
		toggleWakeWordButton.textContent = "Enable Wake Word Mode";
		toggleWakeWordButton.classList.remove('bg-red-500');
		toggleWakeWordButton.classList.add('bg-blue-500');

		if (wakeWordRecognition) {
			wakeWordRecognition.stop();
		}
		hideWaveform();

		// speakFeedback("Wake word mode deactivated.");
	}





	function handleTopLevelCommand(command) {
		clearTimeout(inactivityTimer);
		if (command.includes("computer")) {
			wakeWordState = 'menu';
			inactivityCount = 0;
			speakAndListen("What would you like to do? Say the MODE.", handleMenuCommand);
		} else if (command.includes("goodbye")) {
			deactivateWakeWordMode();
		} else {
			inactivityCount++;
			if (inactivityCount >= 2) {
				speakFeedback(" ");
				deactivateWakeWordMode();
			} else {
				if (isWakeWordModeActive) {
					speakAndListen(" ", handleTopLevelCommand);
				} else {
					handleTopLevelCommand("");
				}
			}
		}
		startInactivityTimer();
	}




	function startInactivityTimer() {
		inactivityTimer = setTimeout(() => {
			handleTopLevelCommand('');
		}, 15000);
	}





	function handleMenuCommand(command) {
		if (command.includes("gmail")) {
			console.log("Gmail command received");

			// Check if an access token already exists in localStorage
			const accessToken = localStorage.getItem('gmail_access_token');
			if (accessToken) {
				console.log("Using existing access token.");
				wakeWordState = 'gmail';  // Set the state to Gmail mode
				speakFeedback("Gmail ready. Starting to read your emails.", () => {
					startReadingEmails();  // Directly start reading emails
				});
			} else {
				console.log("No access token found. Initiating Gmail authentication.");
				wakeWordState = 'processing';
				speakFeedback("Initiating Gmail authentication. Please authorize the app in the popup window.", () => {
					initiateGmailAuth();
				});
			}
		} else if (command.includes("chat")) {
			wakeWordState = 'prompt';
			currentPrompt = '';
			queryType.value = "chat";
			modelType.value = "worker_node";
			modelSelect.value = "2070sLABCHAT";
			speakAndListen("Chat mode. ", handlePromptInput);
		} else if (command.includes("vision")) {
			wakeWordState = 'processing';
			queryType.value = "vision";
			modelType.value = "worker_node";
			updateModelSelect();
			hideWaveform();
			showStaticWaveform();
			callWebcamVisionRoutine();
		} else if (command.includes("imagine")) {
			wakeWordState = 'prompt';
			currentPrompt = '';
			queryType.value = "imagine";
			modelType.value = "worker_node";
			updateModelSelect();
			speakAndListen("Imagine mode. ", handlePromptInput);
		} else if (command.includes("weather")) {
			wakeWordState = 'processing';
			getWeatherData()
				.then(weather => {
					const location = weather.state ? `${weather.city}, ${weather.state}` : weather.city;
					const weatherMessage = `The current weather in ${location} is ${weather.description}. The temperature is ${weather.temperature} degrees Fahrenheit. Humidity is ${weather.humidity}% and wind speed is ${weather.windSpeed} miles per hour.`;
					speakFeedback(weatherMessage, () => {
						deactivateWakeWordMode();
					});
				})
				.catch(error => {
					speakFeedback("I'm sorry, I couldn't get the weather information. " + error, () => {
						deactivateWakeWordMode();
					});
				});
		} else if (command.includes("goodbye")) {
			deactivateWakeWordMode();
		} else {
			speakAndListen("Say your MODE now.", handleMenuCommand);
		}
	}



	function startGmailCommandLoop() {
		console.log("Starting Gmail command loop");
    
		if (!wakeWordRecognition) {
			console.error("Speech recognition not initialized");
			return;
		}

		// Reset command attempts
		gmailCommandAttempts = 0;

		wakeWordRecognition.onresult = function(event) {
			const last = event.results.length - 1;
			const command = event.results[last][0].transcript.trim().toLowerCase();
			console.log("Raw command heard in Gmail loop:", command);

			if (command && command.length > 0) {
				if (command.includes("list") || command.includes("labels") || 
					command.includes("sign out") || command.includes("signout") ||
					command.includes("exit") || command.includes("quit")) {
					handleGmailCommands(command);
				} else {
					console.log("Unrecognized command in Gmail mode:", command);
					speakFeedback("mail", () => {
						startGmailCommandLoop();
					});
				}
			} else {
				console.error('Invalid or undefined command in Gmail loop.');
				gmailCommandAttempts++;
				if (gmailCommandAttempts < MAX_GMAIL_COMMAND_ATTEMPTS) {
					speakFeedback("I didn't catch that. Please try again.", startGmailCommandLoop);
				} else {
					speakFeedback("I'm having trouble understanding. Exiting Gmail mode.", () => {
						wakeWordState = 'listening';
						handleTopLevelCommand("computer");
					});
				}
			}
		};

		wakeWordRecognition.onend = function() {
			console.log("Speech recognition ended in Gmail loop");
			isListening = false;
			if (wakeWordState === 'gmail') {
				console.log("Attempting to restart speech recognition in Gmail loop");
				setTimeout(() => {
					if (!isListening) {
						startGmailCommandLoop();
					}
				}, 100);
			}
		};

		wakeWordRecognition.onerror = function(event) {
			console.error("Speech recognition error in Gmail loop:", event.error);
			isListening = false;
			if (event.error === 'no-speech') {
				console.log("No speech detected, restarting Gmail command loop");
				setTimeout(() => {
					if (!isListening) {
						startGmailCommandLoop();
					}
				}, 100);
			} else {
				speakFeedback("There was an error with speech recognition. Exiting Gmail mode.", () => {
					wakeWordState = 'listening';
					handleTopLevelCommand("computer");
				});
			}
		};

		try {
			if (isListening) {
				wakeWordRecognition.stop();
			}
			setTimeout(() => {
				wakeWordRecognition.start();
				isListening = true;
				console.log("Speech recognition started successfully in Gmail mode");
				speakFeedback("mail.");
			}, 100);
		} catch (error) {
			console.error("Error managing speech recognition in Gmail mode:", error);
			isListening = false;
			speakFeedback("There was an error in Gmail mode. Please try again.", () => {
				wakeWordState = 'listening';
				handleTopLevelCommand("computer");
			});
		}
	}

	function resetGmailModeState() {
		// Ensure that Gmail mode state is fully reset, similar to WWM level state reset logic
		wakeWordState = 'listening';  // Reset to listening mode
		isListening = false;
		isRestarting = false;
		console.log("Gmail mode state reset.");
	}






	async function getWeatherData() {
		return new Promise((resolve, reject) => {
			if ("geolocation" in navigator) {
				navigator.geolocation.getCurrentPosition(async function(position) {
					const lat = position.coords.latitude;
					const lon = position.coords.longitude;
					const apiKey = 'topsecretinfo'; // Replace with your actual API key
					const apiUrl = `https://api.openweathermap.org/data/3.0/onecall?lat=${lat}&lon=${lon}&exclude=minutely,hourly,daily,alerts&units=imperial&appid=${apiKey}`;
					const geoApiUrl = `https://api.openweathermap.org/geo/1.0/reverse?lat=${lat}&lon=${lon}&limit=1&appid=${apiKey}`;

					try {
						const [weatherResponse, geoResponse] = await Promise.all([
							fetch(apiUrl),
							fetch(geoApiUrl)
						]);
						const weatherData = await weatherResponse.json();
						const geoData = await geoResponse.json();

						if (weatherData.current && geoData.length > 0) {
							const weather = {
								temperature: Math.round(weatherData.current.temp),
								description: weatherData.current.weather[0].description,
								humidity: weatherData.current.humidity,
								windSpeed: Math.round(weatherData.current.wind_speed),
								city: geoData[0].name,
								state: geoData[0].state
							};
							resolve(weather);
						} else {
							reject("Unable to parse weather or location data");
						}
					} catch (error) {
						reject("Error fetching weather data: " + error);
					}
				}, function(error) {
					reject("Geolocation error: " + error.message);
				});
			} else {
				reject("Geolocation is not supported by this browser.");
			}
		});
	}






function handlePromptInput(input) {
    clearTimeout(promptInactivityTimer);
    if (input.includes("computer")) {
        // Disable wake word mode before processing the query
        deactivateWakeWordMode();
        
        wakeWordState = 'processing';
        hideWaveform();
        showStaticWaveform();
        submitQuery(currentPrompt.trim());
        promptInactivityCount = 0;
    } else if (input.includes("backspace")) {
        currentPrompt = '';
        promptInput.value = '';
        promptInactivityCount = 0;
        speakAndListen("Prompt erased. ", handlePromptInput);
    } else if (input.trim() === '') {
        promptInactivityCount++;
        if (promptInactivityCount >= 2) {
            wakeWordState = 'listening';
            promptInactivityCount = 0;
            handleTopLevelCommand("computer");
        } else {
            speakAndListen(" ", handlePromptInput);
        }
    } else {
        currentPrompt += ' ' + input;
        promptInput.value = currentPrompt.trim();
        promptInactivityCount = 0;
        speakAndListen(". ", handlePromptInput);
    }
    startPromptInactivityTimer();
}

function startPromptInactivityTimer() {
    promptInactivityTimer = setTimeout(() => {
        handlePromptInput('');
    }, 15000);
}






	function speakAndListen(message, callback) {
		speakFeedback(message, () => {
			if (isWakeWordModeActive) {
				enableSpeechRecognition(callback); // Re-enable listening after speaking
			}
		});
	}



	function disableSpeechRecognition() {
		if (wakeWordRecognition && isListening) {
			wakeWordRecognition.stop();
			isListening = false;
		}
		hideWaveform();
	}
	
	
	
	function enableSpeechRecognition(callback) {
		console.log("Attempting to enable speech recognition. Current state: isListening =", isListening);

		if (isListening) {
			console.log("Speech recognition is already active.");
			if (callback) callback();
			return;
		}

		if (wakeWordRecognition) {
			wakeWordRecognition.onresult = function(event) {
				const last = event.results.length - 1;
				const command = event.results[last][0].transcript.trim().toLowerCase();
				console.log("Heard:", command);
				if (callback) callback(command);
			};

			try {
				wakeWordRecognition.start();
				isListening = true;
				console.log("Speech recognition started.");
			} catch (error) {
				console.error("Error starting speech recognition:", error);
			}
		}

		setTimeout(() => showWaveform(), 500); // 500ms delay to avoid tight loops
	}




	function restartWakeWordRecognition() {
		console.log("Restarting wake word recognition");
		setTimeout(() => enableSpeechRecognition(), 500);  
	}




    function showWaveform() {
        if (audioWaveform) {
            audioWaveform.style.display = 'block';
        }
        drawWaveform();
    }

    function hideWaveform() {
        if (audioWaveform) {
            audioWaveform.style.display = 'none';
        }
        if (typeof animationId !== 'undefined' && animationId) {
            cancelAnimationFrame(animationId);
        }
    }

    function showStaticWaveform() {
        if (audioWaveform) {
            const ctx = audioWaveform.getContext('2d');
            const width = audioWaveform.width;
            const height = audioWaveform.height;
            
            ctx.clearRect(0, 0, width, height);
            ctx.beginPath();
            
            for (let x = 0; x < width; x++) {
                const y = height / 2 + Math.sin((x / width) * Math.PI * 2) * (height / 4);
                if (x === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    }





	async function callWebcamVisionRoutine() {
		speakFeedback("Accessing webcam for vision processing.", async () => {
			const video = await setupCamera();
			if (video) {
				showStaticWaveform();
				await new Promise(resolve => setTimeout(resolve, 1000)); // Give the camera a moment to adjust
				const imageData = await captureImage(video);
				stopCamera();
				document.body.removeChild(video); // Remove the video element from the DOM
            
				// Display the captured image
				displayCapturedImage(imageData);
            
				await processVisionQuery(imageData);
			} else {
				wakeWordState = 'listening';
				handleTopLevelCommand("computer");
			}
		});
	}









	async function processVisionQuery(imageData) {
		const query = {
			type: 'submit_query',
			query: {
				prompt: "Describe this image in detail",
				query_type: "vision",
				model_type: "worker_node",
				model_name: "default_vision_model",
				image: imageData.split(',')[1] // Remove the "data:image/jpeg;base64," part
			}
		};

		sendToWebSocket(query);
	}










    function handleWakeWordCommand(action, fullCommand) {
        console.log("Handling command:", action, fullCommand);
        switch (action) {
            case "chat":
                queryType.value = "chat";
                modelType.value = "worker_node";
                modelSelect.value = "2070sLABCHAT";
                promptInput.value = fullCommand.split("chat")[1].replace(/\s*submit\s*$/i, "").trim();
                break;
            case "vision":
                queryType.value = "vision";
                modelType.value = "worker_node";
                updateModelSelect();
                break;
            case "imagine":
                queryType.value = "imagine";
                modelType.value = "worker_node";
                updateModelSelect();
                promptInput.value = fullCommand.split("imagine")[1].replace(/\s*submit\s*$/i, "").trim();
                break;
        }

        updateModelTypeOptions();
        handleModelTypeChange();

        const statusMessage = `Ready to ${action}. Your prompt is: "${promptInput.value}". Say "submit" to send the query, or modify the prompt manually.`;
        speakAndListen(statusMessage, handlePromptInput);
    }

    function handleSubmitQuery(event) {
        event.preventDefault();
        console.log("handleSubmitQuery function called");
        if (validateForm()) {
            submitQuery();
        }
    }










	function submitQuery(prompt) {
		console.log('submitQuery function called');
    
		if (!prompt) {
			prompt = promptInput.value.trim();
		}
		const type = queryType.value;
		const modelTypeValue = modelType.value;
		const modelName = modelSelect.value;

		const query = {
			type: 'submit_query',
			query: {
				prompt: prompt,
				query_type: type,
				model_type: modelTypeValue,
				model_name: modelName
			}
		};

		console.log('Preparing to send query:', query);

		if (wakeWordState !== 'inactive') {
			query.query.model_type = 'speech';
		}

		if (type === 'vision' && imageUpload.files[0]) {
			sendImageChunks(imageUpload.files[0], query);
		} else if (type === 'speech' && audioChunks.length > 0) {
			sendAudioQuery(query);
		} else {
			sendToWebSocket(query);
		}

		promptInput.value = '';
		if (imageUpload) imageUpload.value = '';
		if (imagePreview) imagePreview.style.display = 'none';
		audioChunks = []; // Clear audio chunks after sending

		speakFeedback("Query submitted. Processing your request.");
	}

















    function sendImageChunks(file, query) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const imageData = e.target.result.split(',')[1]; // Get base64 data
            const imageId = Date.now().toString(); // Use timestamp as unique ID
            const totalChunks = Math.ceil(imageData.length / CHUNK_SIZE);

            for (let i = 0; i < totalChunks; i++) {
                const start = i * CHUNK_SIZE;
                const end = Math.min((i + 1) * CHUNK_SIZE, imageData.length);
                const chunk = imageData.slice(start, end);

                sendToWebSocket({
                    type: 'vision_chunk',
                    chunk_id: i,
                    total_chunks: totalChunks,
                    chunk_data: chunk,
                    image_id: imageId,
                    prompt: query.query.prompt,
                    query_type: query.query.query_type,
                    model_type: query.query.model_type,
                    model_name: query.query.model_name
                });
            }
        };
        reader.readAsDataURL(file);
    }

    function sendAudioQuery(query) {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64Audio = reader.result.split(',')[1];
            query.query.audio = base64Audio;
            sendToWebSocket(query);
        };
        reader.readAsDataURL(audioBlob);
    }










	function handleQueryResult(result, processingTime, cost, resultType) {
		const resultElement = document.createElement('div');
    
		if (resultType === 'image') {
			const img = document.createElement('img');
			img.src = 'data:image/png;base64,' + result;
			img.alt = 'Generated Image';
			img.className = 'max-w-full h-auto';
			resultElement.appendChild(img);
			if (wakeWordState !== 'inactive') {
				speakFeedback("Image generated successfully.", deactivateWakeWordMode);
			}
		} else if (resultType === 'audio') {
			handleSpeechResult(result);
		} else if (result) {
			const formattedResult = result.replace(/```(\w+)?\n([\s\S]*?)```/g, function(match, language, code) {
				return `<pre><code class="language-${language || ''}">${escapeHtml(code.trim())}</code></pre>`;
			});
        
			resultElement.innerHTML = `<div class="result-content">${formattedResult}</div>`;
        
			if (wakeWordState !== 'inactive') {
				if (checkForWeapons(result)) {
					speakFeedback("WEAPON DETECTED - FACILITY LOCKED DOWN - POLICE RESPONDING", deactivateWakeWordMode);
				} else {
					speakFeedback(result);
				}
			}
		}
    
		resultElement.innerHTML += `
			<p><strong>Processing Time:</strong> ${processingTime.toFixed(2)}s</p>
			<p><strong>Cost:</strong> $${cost.toFixed(4)}</p>
		`;
		resultElement.className = 'mb-4 p-4 bg-gray-100 rounded';
		results.prepend(resultElement);
		updateCumulativeCosts(currentUser);
    
		if (typeof Prism !== 'undefined') {
			resultElement.querySelectorAll('pre code').forEach((block) => {
				Prism.highlightElement(block);
			});
		}
	}



	function checkForWeapons(visionResponse) {
		const weaponKeywords = ['knife', 'gun', 'weapon', 'firearm', 'blade'];
		const lowercaseResponse = visionResponse.toLowerCase();
    
		for (const keyword of weaponKeywords) {
			if (lowercaseResponse.includes(keyword)) {
				return true;
			}
		}
    
		return false;
	}











    function handleQueryTypeChange() {
        if (queryType.value === 'vision') {
            imageUpload.style.display = 'block';
            voiceInputButton.style.display = 'none';
            promptInput.disabled = false;
        } else if (queryType.value === 'speech') {
            imageUpload.style.display = 'none';
            voiceInputButton.style.display = 'inline-block';
            promptInput.disabled = true;
        } else {
            imageUpload.style.display = 'none';
            voiceInputButton.style.display = 'none';
            promptInput.disabled = false;
        }
        updateModelTypeOptions();
    }

    function handleModelTypeChange() {
        updateModelSelect();
    }

    function updateModelTypeOptions() {
        modelType.innerHTML = '';

        if (queryType.value === 'chat' || queryType.value === 'speech') {
            addOption(modelType, 'worker_node', 'Worker Node');
            addOption(modelType, 'huggingface', 'Hugging Face');
            addOption(modelType, 'claude', 'Claude');
        } else if (queryType.value === 'vision') {
            addOption(modelType, 'worker_node', 'Worker Node');
            addOption(modelType, 'huggingface', 'Hugging Face');
        } else if (queryType.value === 'imagine') {
            addOption(modelType, 'worker_node', 'Worker Node');
        }

        handleModelTypeChange();
    }

    function updateModelSelect() {
        modelSelect.innerHTML = '';
        const selectedModelType = modelType.value;

        if (selectedModelType === 'huggingface') {
            Object.values(huggingFaceModels).forEach(model => {
                addOption(modelSelect, model.name, model.name);
            });
        } else if (selectedModelType === 'worker_node') {
            Object.values(aiWorkers).forEach(worker => {
                if (worker.type === queryType.value || (queryType.value === 'speech' && worker.type === 'chat')) {
                    addOption(modelSelect, worker.name, worker.name);
                }
            });
        } else if (selectedModelType === 'claude') {
            addOption(modelSelect, 'claude-2.1', 'Claude-2.1');
        }
    }

    function addOption(selectElement, value, text) {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = text;
        selectElement.appendChild(option);
    }

    function handleImageUpload(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                imagePreview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    }

    function updateQueueStatus(depth, total) {
        const percentage = (depth / total) * 100;
        queueThermometer.style.width = `${percentage}%`;
        queueThermometer.textContent = `Queue: ${depth}`;
    }

    function displayPreviousQueries(queries) {
        previousQueries.innerHTML = '';
        if (queries.length === 0) {
            previousQueries.innerHTML = '<p>No previous queries</p>';
            return;
        }
        queries.forEach(query => {
            const queryElement = document.createElement('div');
            queryElement.innerHTML = `
                <p><strong>Prompt:</strong> ${escapeHtml(query.prompt)}</p>
                <p><strong>Type:</strong> ${query.query_type}</p>
                <p><strong>Model:</strong> ${query.model_type} - ${query.model_name}</p>
                <p><strong>Processing Time:</strong> ${query.processing_time.toFixed(2)}s</p>
                <p><strong>Cost:</strong> $${query.cost.toFixed(4)}</p>
                <p><strong>Timestamp:</strong> ${new Date(query.timestamp).toLocaleString()}</p>
            `;
            queryElement.className = 'mb-4 p-4 bg-gray-100 rounded';
            previousQueries.appendChild(queryElement);
        });
    }

    function updateSystemStats(stats) {
        if (systemStats) {
            systemStats.innerHTML = `
                <p><strong>Total Queries:</strong> ${stats.total_queries}</p>
                <p><strong>Total Processing Time:</strong> ${stats.total_processing_time.toFixed(2)}s</p>
                <p><strong>Total Cost:</strong> $${stats.total_cost.toFixed(4)}</p>
                <p><strong>Last Updated:</strong> ${new Date(stats.last_updated).toLocaleString()}</p>
            `;
            updateSystemStatsChart(stats);
        }
    }

    function updateUserStats(users) {
        userList.innerHTML = '';
        users.forEach(user => {
            const userElement = document.createElement('div');
            userElement.innerHTML = `
                <p><strong>Nickname:</strong> ${escapeHtml(user.nickname)}</p>
                <p><strong>GUID:</strong> ${user.guid}</p>
                <p><strong>Total Query Time:</strong> ${user.total_query_time.toFixed(2)}s</p>
                <p><strong>Total Cost:</strong> $${user.total_cost.toFixed(4)}</p>
                <p><strong>Banned:</strong> ${user.is_banned ? 'Yes' : 'No'}</p>
                <button class="ban-user" data-guid="${user.guid}">${user.is_banned ? 'Unban' : 'Ban'}</button>
                <button class="terminate-query" data-guid="${user.guid}">Terminate Query</button>
            `;
            userElement.className = 'mb-4 p-4 bg-gray-100 rounded';
            userList.appendChild(userElement);
        });

        document.querySelectorAll('.ban-user').forEach(button => {
            button.addEventListener('click', () => {
                const guid = button.getAttribute('data-guid');
                const isBanned = button.textContent === 'Unban';
                sendToWebSocket({
                    type: isBanned ? 'unban_user' : 'ban_user',
                    user_guid: guid
                });
            });
        });

        document.querySelectorAll('.terminate-query').forEach(button => {
            button.addEventListener('click', () => {
                const guid = button.getAttribute('data-guid');
                sendToWebSocket({
                    type: 'terminate_query',
                    user_guid: guid
                });
            });
        });
    }

    function updateWorkerHealth(healthData) {
        const workerHealthElement = document.getElementById('worker-health');
        workerHealthElement.innerHTML = '';
        healthData.forEach(worker => {
            const workerElement = document.createElement('div');
            workerElement.innerHTML = `
                <p><strong>${escapeHtml(worker.name)}</strong></p>
                <p>Health Score: ${worker.health_score.toFixed(2)}</p>
                <p>Status: ${worker.is_blacklisted ? 'Blacklisted' : 'Active'}</p>
                <p>Last Active: ${new Date(worker.last_active).toLocaleString()}</p>
            `;
            workerElement.className = `mb-2 p-2 rounded ${getWorkerStatusClass(worker)}`;
            workerHealthElement.appendChild(workerElement);
        });
    }

    function updateWorkerList(workers) {
        aiWorkers = workers.reduce((acc, worker) => {
            acc[worker.name] = worker;
            return acc;
        }, {});

        workerList.innerHTML = '';
        workers.forEach(worker => {
            const workerElement = document.createElement('div');
            workerElement.innerHTML = `
                <p><strong>Name:</strong> ${escapeHtml(worker.name)}</p>
                <p><strong>Address:</strong> ${escapeHtml(worker.address)}</p>
                <p><strong>Type:</strong> ${worker.type}</p>
                <p><strong>Health Score:</strong> ${worker.health_score.toFixed(2)}</p>
                <p><strong>Status:</strong> ${worker.is_blacklisted ? 'Blacklisted' : 'Active'}</p>
                <p><strong>Last Active:</strong> ${new Date(worker.last_active).toLocaleString()}</p>
                <button class="remove-worker" data-name="${escapeHtml(worker.name)}">Remove</button>
            `;
            workerElement.className = `mb-4 p-4 rounded ${getWorkerStatusClass(worker)}`;
            workerList.appendChild(workerElement);
        });

        document.querySelectorAll('.remove-worker').forEach(button => {
            button.addEventListener('click', () => {
                const name = button.getAttribute('data-name');
                sendToWebSocket({
                    type: 'remove_worker',
                    worker_name: name
                });
            });
        });
        updateModelSelect();
    }

    function getWorkerStatusClass(worker) {
        if (worker.is_blacklisted) return 'bg-black text-white';
        if (worker.health_score < 50) return 'bg-red-200';
        if (worker.health_score < 80) return 'bg-yellow-200';
        return 'bg-green-200';
    }

    function updateHuggingFaceModelList(models) {
        huggingFaceModels = models.reduce((acc, model) => {
            acc[model.name] = model;
            return acc;
        }, {});

        huggingFaceModelList.innerHTML = '';
        models.forEach(model => {
            const modelElement = document.createElement('div');
            modelElement.innerHTML = `
                <p><strong>Name:</strong> ${escapeHtml(model.name)}</p>
                <p><strong>Type:</strong> ${model.type}</p>
                <button class="remove-huggingface-model" data-name="${escapeHtml(model.name)}">Remove</button>
            `;
            modelElement.className = 'mb-4 p-4 bg-gray-100 rounded';
            huggingFaceModelList.appendChild(modelElement);
        });

        document.querySelectorAll('.remove-huggingface-model').forEach(button => {
            button.addEventListener('click', () => {
                const name = button.getAttribute('data-name');
                sendToWebSocket({
                    type: 'remove_huggingface_model',
                    model_name: name
                });
            });
        });

        updateModelSelect();
    }

    function updateActiveUsers(users) {
        activeUsersTable.innerHTML = '';
        users.forEach(user => {
            const row = activeUsersTable.insertRow();
            const cellUser = row.insertCell(0);
            const cellActions = row.insertCell(1);
            
            cellUser.textContent = escapeHtml(user);
            
            const banButton = document.createElement('button');
            banButton.textContent = 'Ban';
            banButton.className = 'bg-red-500 text-white px-2 py-1 rounded mr-2';
            banButton.onclick = () => banUser(user);
            
            const terminateButton = document.createElement('button');
            terminateButton.textContent = 'Terminate Queue';
            terminateButton.className = 'bg-yellow-500 text-white px-2 py-1 rounded';
            terminateButton.onclick = () => terminateUserQueue(user);
            
            cellActions.appendChild(banButton);
            cellActions.appendChild(terminateButton);
        });
    }

    function banUser(user) {
        if (confirm(`Are you sure you want to ban user ${user}?`)) {
            sendToWebSocket({
                type: 'ban_user',
                user_guid: user
            });
        }
    }

    function unbanUser(user) {
        if (confirm(`Are you sure you want to unban user ${user}?`)) {
            sendToWebSocket({
                type: 'unban_user',
                user_guid: user
            });
        }
    }

    function terminateUserQueue(user) {
        if (confirm(`Are you sure you want to terminate all queued tasks for user ${user}?`)) {
            sendToWebSocket({
                type: 'terminate_query',
                user_guid: user
            });
        }
    }

    function sendSysopMessage() {
        const message = sysopMessageInput.value.trim();
        if (message) {
            sendToWebSocket({
                type: 'sysop_message',
                message: message
            });
            sysopMessageInput.value = '';
        }
    }

    function handleUserBanned(guid) {
        if (currentUser && currentUser.guid === guid) {
            displayError("You have been banned from using this service.");
            disableUserInterface();
        }
        sendToWebSocket({ type: 'get_stats' }); // Refresh user list for sysop
    }

    function handleUserUnbanned(guid) {
        if (currentUser && currentUser.guid === guid) {
            displayStatus("Your ban has been lifted. You can now use the service again.");
            enableUserInterface();
        }
        sendToWebSocket({ type: 'get_stats' }); // Refresh user list for sysop
    }

    function handleQueryTerminated(guid) {
        if (currentUser && currentUser.guid === guid) {
            displayStatus("Your query was terminated by a sysop.");
        }
        sendToWebSocket({ type: 'get_stats' }); // Refresh queue status
    }

    function displaySysopMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.textContent = `Sysop Message: ${message}`;
        messageElement.className = 'mb-4 p-4 bg-yellow-100 rounded';
        results.prepend(messageElement);
    }

    function displayError(message) {
        const errorElement = document.createElement('div');
        errorElement.textContent = `Error: ${message}`;
        errorElement.className = 'mb-4 p-4 bg-red-100 rounded';
        results.prepend(errorElement);
    }

    function displayStatus(message) {
        const statusElement = document.createElement('div');
        statusElement.textContent = message;
        statusElement.className = 'mb-4 p-4 bg-blue-100 rounded';
        results.prepend(statusElement);
    }

    function disableUserInterface() {
        promptInput.disabled = true;
        submitQueryButton.disabled = true;
        queryType.disabled = true;
        modelType.disabled = true;
        modelSelect.disabled = true;
        imageUpload.disabled = true;
        voiceInputButton.disabled = true;
    }

    function enableUserInterface() {
        promptInput.disabled = false;
        submitQueryButton.disabled = false;
        queryType.disabled = false;
        modelType.disabled = false;
        modelSelect.disabled = false;
        imageUpload.disabled = false;
        voiceInputButton.disabled = false;
    }

    function sendToWebSocket(data) {
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify(data));
        } else {
            console.error('WebSocket is not open. ReadyState:', socket.readyState);
            displayError("WebSocket connection is not open. Please try again.");
            scheduleReconnection();
        }
    }

    function validateForm() {
        let isValid = true;
        if (queryType.value === 'speech' && audioChunks.length === 0) {
            displayError('Please record your voice query before submitting.');
            isValid = false;
        } else if (queryType.value !== 'speech' && promptInput.value.trim() === '') {
            displayError('Please enter a prompt');
            isValid = false;
        }
        if (queryType.value === 'vision' && imageUpload && !imageUpload.files[0]) {
            displayError('Please upload an image for vision queries');
            isValid = false;
        }
        return isValid;
    }

    function updateCumulativeCosts(user) {
        if (cumulativeCosts) {
            cumulativeCosts.innerHTML = `
                <p><strong>Total Query Time:</strong> ${user.total_query_time.toFixed(2)}s</p>
                <p><strong>Total Cost:</strong> $${user.total_cost.toFixed(4)}</p>
            `;
        }
    }

    function clearResults() {
        if (results) {
            results.innerHTML = '';
        }
    }

    function updateConnectionStatus(isConnected) {
        if (connectionStatus) {
            connectionStatus.textContent = isConnected ? 'Connected' : 'Disconnected';
            connectionStatus.className = isConnected ? 'text-green-500' : 'text-red-500';
        }
    }

    function updateSystemStatsChart(stats) {
        const ctx = document.getElementById('system-stats-chart');
        if (ctx && typeof Chart !== 'undefined') {
            if (!ctx.chart) {
                ctx.chart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Total Queries',
                            data: [],
                            borderColor: 'rgb(75, 192, 192)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });
            }

            const chart = ctx.chart;
            chart.data.labels.push(new Date().toLocaleTimeString());
            chart.data.datasets[0].data.push(stats.total_queries);

            if (chart.data.labels.length > 10) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
            }

            chart.update();
        }
    }

    function escapeHtml(unsafe) {
        return unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }

    // Voice recording functions
    function toggleVoiceRecording() {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    }

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };
            mediaRecorder.onstop = sendVoiceQuery;
            mediaRecorder.start();
            isRecording = true;
            voiceInputButton.textContent = 'Stop Recording';
            voiceInputButton.classList.add('bg-red-500');
            voiceInputButton.classList.remove('bg-blue-500');
        } catch (err) {
            console.error('Error accessing microphone:', err);
            displayError('Error accessing microphone. Please ensure you have given permission to use the microphone.');
        }
    }

    function stopRecording() {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            isRecording = false;
            voiceInputButton.textContent = 'Start Voice Input';
            voiceInputButton.classList.remove('bg-red-500');
            voiceInputButton.classList.add('bg-blue-500');
        }
    }

    function sendVoiceQuery() {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64Audio = reader.result.split(',')[1];
            sendToWebSocket({
                type: 'speech_to_text',
                audio: base64Audio
            });
        };
        reader.readAsDataURL(audioBlob);
    }

    function toggleSpeechOutput() {
        speechOutputEnabled = speechOutputCheckbox.checked;
    }

    function handleTranscriptionResult(text) {
        promptInput.value = text;
        displayStatus('Voice input transcribed. You can now submit the query.');
    }







function handleSpeechResult(audioBase64) {
    const audioSrc = 'data:audio/webm;base64,' + audioBase64;
    audioQueue.push(audioSrc);
    if (!isAudioPlaying) {
        playNextAudio();
    }

    // Add speech recognition for "SHUT UP" command
    if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = false;

        recognition.onresult = function(event) {
            const last = event.results.length - 1;
            const command = event.results[last][0].transcript.trim().toLowerCase();
            if (command.includes("shut up")) {
                stopAllAudio();
                deactivateWakeWordMode();
            }
        };

        recognition.start();
    }
}

function stopAllAudio() {
    audioQueue = [];
    if (persistentAudio) {
        persistentAudio.pause();
        persistentAudio.currentTime = 0;
    }
    isAudioPlaying = false;
}


	function playNextAudio() {
		if (audioQueue.length > 0) {
			isAudioPlaying = true;
			const audioSrc = audioQueue.shift();
			const audio = new Audio(audioSrc);

			// Add the audio element to the DOM
			document.body.appendChild(audio);

			audio.onended = function() {
				// Remove the audio element from the DOM when playback is complete
				document.body.removeChild(audio);
				isAudioPlaying = false;
				playNextAudio(); // Play the next audio in the queue, if any
			};

			audio.onerror = function(error) {
				console.error('Error playing audio:', error);
				displayError('Error playing audio response. Please check your audio settings.');
				// Remove the audio element from the DOM in case of error
				document.body.removeChild(audio);
				isAudioPlaying = false;
				playNextAudio(); // Try to play the next audio in the queue, if any
			};

			// Use a promise to handle the play() method
			audio.play().then(() => {
				console.log('Audio playback started successfully');
				displayStatus('Audio response is playing.');
			}).catch(error => {
				console.error('Error starting audio playback:', error);
				displayError('Error playing audio response. Please check your audio settings.');
				// Remove the audio element from the DOM in case of error
				document.body.removeChild(audio);
				isAudioPlaying = false;
				playNextAudio(); // Try to play the next audio in the queue, if any
			});
		}
	}


	function handleAudioError(error) {
		console.error('Error playing audio:', error);
		displayError('Error playing audio response. Please check your audio settings.');
		isAudioPlaying = false;
		playNextAudio(); // Try to play the next audio in the queue, if any
	}



    // Initialize the application
    function init() {
        // updateQueryTypeOptions();
        handleQueryTypeChange();
        updateModelSelect();
		setupAudioHandling();

        if (currentUser && currentUser.is_sysop) {
            startPeriodicUpdates();
        }
        loadPreferences();
    }

    // Periodic updates for sysop
    function startPeriodicUpdates() {
        if (currentUser && currentUser.is_sysop) {
            // Clear any existing interval
            if (window.statsUpdateInterval) {
                clearInterval(window.statsUpdateInterval);
            }
            // Set new interval
            window.statsUpdateInterval = setInterval(() => {
                sendToWebSocket({ type: 'get_stats' });
            }, 30000); // Update every 30 seconds
        }
    }

    // Add event listeners for forms in the sysop panel
    const addWorkerForm = document.getElementById('add-worker-form');
    if (addWorkerForm) {
        addWorkerForm.addEventListener('submit', function(event) {
            event.preventDefault();
            const name = document.getElementById('worker-name').value;
            const address = document.getElementById('worker-address').value;
            const type = document.getElementById('worker-type').value;
            sendToWebSocket({
                type: 'add_worker',
                worker: { name, address, type }
            });
            this.reset();
        });
    }

    const addHuggingfaceModelForm = document.getElementById('add-huggingface-model-form');
    if (addHuggingfaceModelForm) {
        addHuggingfaceModelForm.addEventListener('submit', function(event) {
            event.preventDefault();
            const name = document.getElementById('huggingface-model-name').value;
            const type = document.getElementById('huggingface-model-type').value;
            sendToWebSocket({
                type: 'add_huggingface_model',
                model: { name, type }
            }); 
            this.reset();
        });
    }

    // Handle visibility change to reconnect if needed
    document.addEventListener("visibilitychange", function() {
        if (!document.hidden && (!socket || socket.readyState !== WebSocket.OPEN)) {
            connectWebSocket();
        }
    });

    // Add responsive design adjustments
    function adjustLayoutForScreenSize() {
        const mainContent = document.querySelector('main');
        if (mainContent) {
            if (window.innerWidth < 768) {
                mainContent.classList.remove('grid', 'grid-cols-2', 'gap-4');
                mainContent.classList.add('flex', 'flex-col');
            } else {
                mainContent.classList.add('grid', 'grid-cols-2', 'gap-4');
                mainContent.classList.remove('flex', 'flex-col');
            }
        }
    }

    window.addEventListener('resize', adjustLayoutForScreenSize);
    adjustLayoutForScreenSize(); // Call once at init to set initial layout

    // Add ARIA attributes for better accessibility
    if (submitQueryButton) submitQueryButton.setAttribute('aria-label', 'Submit Query');
    if (imageUpload) imageUpload.setAttribute('aria-label', 'Upload Image for Vision Query');
    if (sysopMessageInput) sysopMessageInput.setAttribute('aria-label', 'Sysop Message Input');
    if (sendSysopMessageButton) sendSysopMessageButton.setAttribute('aria-label', 'Send Sysop Message');
    if (voiceInputButton) voiceInputButton.setAttribute('aria-label', 'Toggle Voice Recording');

    // Function to handle file drag and drop
    function handleDragDrop() {
        const dropZone = document.getElementById('image-drop-zone');
        if (!dropZone) return;

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, unhighlight, false);
        });

        function highlight() {
            dropZone.classList.add('bg-blue-100');
        }

        function unhighlight() {
            dropZone.classList.remove('bg-blue-100');
        }

        dropZone.addEventListener('drop', handleDrop, false);

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length) {
                imageUpload.files = files;
                handleImageUpload({ target: imageUpload });
            }
        }
    }

    // Initialize drag and drop functionality
    handleDragDrop();

    // Function to update the UI theme
    function updateTheme(isDark) {
        const root = document.documentElement;
        if (isDark) {
            root.classList.add('dark');
        } else {
            root.classList.remove('dark');
        }
    }

    // Check user's preferred color scheme
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        updateTheme(true);
    }

    // Listen for changes in color scheme preference
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
        updateTheme(e.matches);
    });

    // Function to handle keyboard shortcuts
    function handleKeyboardShortcuts(e) {
        // Ctrl/Cmd + Enter to submit query
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            handleSubmitQuery(e);
        }
        // Ctrl/Cmd + L to clear results
        if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
            e.preventDefault();
            clearResults();
        }
        // Ctrl/Cmd + Shift + V to toggle voice input
        if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'V') {
            e.preventDefault();
            toggleVoiceRecording();
        }
        // Ctrl/Cmd + Shift + S to toggle speech output
        if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'S') {
            e.preventDefault();
            speechOutputCheckbox.checked = !speechOutputCheckbox.checked;
            toggleSpeechOutput();
        }
        // Ctrl/Cmd + Shift + W to toggle wake word mode
        if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'W') {
            e.preventDefault();
            toggleWakeWordMode();
        }
    }

    // Add keyboard shortcut listener
    document.addEventListener('keydown', handleKeyboardShortcuts);

    // Function to show a tooltip
    function showTooltip(element, message) {
        const tooltip = document.createElement('div');
        tooltip.textContent = message;
        tooltip.className = 'absolute bg-gray-800 text-white p-2 rounded text-sm z-10';
        element.appendChild(tooltip);
        setTimeout(() => tooltip.remove(), 3000);
    }

    // Add tooltips to important elements
    if (submitQueryButton) {
        submitQueryButton.addEventListener('mouseover', () => showTooltip(submitQueryButton, 'Submit your query (Ctrl/Cmd + Enter)'));
    }
    if (clearResultsButton) {
        clearResultsButton.addEventListener('mouseover', () => showTooltip(clearResultsButton, 'Clear all results (Ctrl/Cmd + L)'));
    }
    if (voiceInputButton) {
        voiceInputButton.addEventListener('mouseover', () => showTooltip(voiceInputButton, 'Start/Stop voice recording (Ctrl/Cmd + Shift + V)'));
    }
    if (speechOutputCheckbox) {
        speechOutputCheckbox.parentElement.addEventListener('mouseover', () => showTooltip(speechOutputCheckbox.parentElement, 'Enable/Disable speech output (Ctrl/Cmd + Shift + S)'));
    }
    if (toggleWakeWordButton) {
        toggleWakeWordButton.addEventListener('mouseover', () => showTooltip(toggleWakeWordButton, 'Enable/Disable wake word mode (Ctrl/Cmd + Shift + W)'));
    }

    // Function to handle errors gracefully
    function handleError(error) {
        console.error('An error occurred:', error);
        displayError('An unexpected error occurred. Please try again or contact support if the problem persists.');
    }

    // Wrap all async functions with error handling
    ['submitQuery', 'handleSubmitQuery', 'handleQueryResult', 'startRecording', 'stopRecording', 'sendVoiceQuery'].forEach(funcName => {
        const original = window[funcName];
        window[funcName] = async function(...args) {
            try {
                await original.apply(this, args);
            } catch (error) {
                handleError(error);
            }
        };
    });

    // Function to save user preferences
    function savePreferences() {
        const preferences = {
            theme: document.documentElement.classList.contains('dark') ? 'dark' : 'light',
            fontSize: document.body.style.fontSize,
            speechOutputEnabled: speechOutputEnabled,
            wakeWordState: wakeWordState
        };
        localStorage.setItem('userPreferences', JSON.stringify(preferences));
    }

    // Function to load user preferences
    function loadPreferences() {
        const savedPreferences = localStorage.getItem('userPreferences');
        if (savedPreferences) {
            const preferences = JSON.parse(savedPreferences);
            updateTheme(preferences.theme === 'dark');
            document.body.style.fontSize = preferences.fontSize || '16px';
            speechOutputEnabled = preferences.speechOutputEnabled || false;
            if (speechOutputCheckbox) {
                speechOutputCheckbox.checked = speechOutputEnabled;
            }
            if (preferences.wakeWordState === 'listening') {
                toggleWakeWordMode();
            }
        }
    }

    // Save preferences when changed
    window.addEventListener('beforeunload', savePreferences);

    // Expose necessary functions to the global scope if needed
    window.RENTAHAL = {
        submitQuery,
        clearResults,
        updateTheme,
        toggleVoiceRecording,
        toggleSpeechOutput,
        toggleWakeWordMode
    };

    // Start the WebSocket connection
    connectWebSocket();
});
