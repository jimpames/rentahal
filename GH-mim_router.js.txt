// MasterIntentMatrix.js - Core cognitive routing system for RENT A HAL
// Drop-in replacement for traditional routing in webgui.py and script.js
// (C) Copyright 2025, The N2NHU Lab for Applied AI
// Designer: J.P. Ames, N2NHU  
// Architect: Claude (Anthropic)
// Released under GPL-3.0 with eternal openness

import sqlite3 from 'sqlite3';
import { EventEmitter } from 'events';

class Intent {
    constructor(name, weight, idealState, threshold = 0.5) {
        this.name = name;
        this.weight = weight;
        this.idealState = idealState;
        this.threshold = threshold;
        this.currentDistance = 0;
        this.activated = false;
        this.sensors = new Map();
        this.subIntents = [];
        this.lastActivation = null;
        this.activationCount = 0;
    }

    calculateDistance(currentState) {
        let distance = 0;
        for (const [key, idealValue] of Object.entries(this.idealState)) {
            const currentValue = currentState[key] || 0;
            distance += Math.pow(idealValue - currentValue, 2);
        }
        this.currentDistance = Math.sqrt(distance);
        return this.currentDistance;
    }

    shouldActivate(sensorData, currentState) {
        const distance = this.calculateDistance(currentState);
        let sensorScore = 0;
        
        for (const [sensorId, threshold] of this.sensors) {
            const value = sensorData.get(sensorId) || 0;
            if (value > threshold) {
                sensorScore += value / threshold;
            }
        }
        
        return sensorScore > this.threshold && distance < this.threshold * 2;
    }
}

class RealityMembrane {
    constructor(decayRate = 0.5, cognitiveTemperature = 1.0) {
        this.intents = new Map();
        this.decayRate = decayRate;
        this.cognitiveTemperature = cognitiveTemperature;
        this.membraneState = {
            temperature: cognitiveTemperature,
            decayRate: decayRate,
            lastUpdate: Date.now()
        };
    }

    addIntent(intent) {
        this.intents.set(intent.name, intent);
    }

    calculateIntentPotential(intentName, currentState) {
        const intent = this.intents.get(intentName);
        if (!intent) return 0;

        const distance = intent.calculateDistance(currentState);
        const potential = intent.weight * Math.exp(-this.decayRate * distance);
        return potential;
    }

    updateSystem(deltaTime) {
        const now = Date.now();
        const dt = (now - this.membraneState.lastUpdate) / 1000;
        
        // Natural decay of unused intents
        for (const intent of this.intents.values()) {
            if (!intent.activated && intent.weight > 1) {
                intent.weight *= Math.exp(-0.1 * dt);
            }
        }
        
        this.membraneState.lastUpdate = now;
    }
}

class CrystallineMemorySystem {
    constructor(dbPath = './rentahal_memory.db') {
        this.db = new sqlite3.Database(dbPath);
        this.initializeDatabase();
    }

    initializeDatabase() {
        this.db.serialize(() => {
            // Waveform memory table
            this.db.run(`CREATE TABLE IF NOT EXISTS waveform_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                intent_name TEXT,
                state_vector TEXT,
                resonance_signature TEXT,
                activation_context TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )`);

            // Intent history table
            this.db.run(`CREATE TABLE IF NOT EXISTS intent_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intent_name TEXT,
                weight REAL,
                activation_time INTEGER,
                success_score REAL,
                context_data TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )`);

            // Reality membrane states
            this.db.run(`CREATE TABLE IF NOT EXISTS membrane_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                temperature REAL,
                decay_rate REAL,
                active_intents TEXT,
                system_state TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )`);

            // CNC G-code memory etchings
            this.db.run(`CREATE TABLE IF NOT EXISTS crystal_etchings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER,
                x_coord REAL,
                y_coord REAL,
                z_coord REAL,
                laser_power INTEGER,
                feed_rate INTEGER,
                etching_data TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(memory_id) REFERENCES waveform_memory(id)
            )`);
        });
    }

    // Store waveform memory with CNC coordinates
    async storeWaveformMemory(intentName, stateVector, context) {
        return new Promise((resolve, reject) => {
            const timestamp = Date.now();
            const resonanceSignature = this.calculateResonanceSignature(stateVector);
            
            this.db.run(
                `INSERT INTO waveform_memory (timestamp, intent_name, state_vector, resonance_signature, activation_context) 
                 VALUES (?, ?, ?, ?, ?)`,
                [timestamp, intentName, JSON.stringify(stateVector), resonanceSignature, JSON.stringify(context)],
                function(err) {
                    if (err) {
                        reject(err);
                        return;
                    }
                    
                    // Generate CNC coordinates for crystalline etching
                    const memoryId = this.lastID;
                    const coordinates = this.generateCNCCoordinates(stateVector, resonanceSignature);
                    
                    this.etchCrystalMemory(memoryId, coordinates).then(() => {
                        resolve(memoryId);
                    });
                }.bind(this)
            );
        });
    }

    calculateResonanceSignature(stateVector) {
        // Create resonance signature for memory recall
        let signature = '';
        for (let i = 0; i < stateVector.length; i++) {
            const value = Math.abs(stateVector[i]);
            signature += Math.floor(value * 255).toString(16).padStart(2, '0');
        }
        return signature.substring(0, 16); // 16-character signature
    }

    generateCNCCoordinates(stateVector, resonanceSignature) {
        const coordinates = [];
        const baseRadius = 10.0; // 10mm base radius
        
        // Convert state vector to 3D coordinates
        for (let i = 0; i < stateVector.length; i += 3) {
            const x = (stateVector[i] || 0) * baseRadius;
            const y = (stateVector[i + 1] || 0) * baseRadius;
            const z = (stateVector[i + 2] || 0) * baseRadius * 0.1; // Shallow etching
            
            const laserPower = Math.floor(Math.abs(x + y) * 10) % 255;
            const feedRate = Math.max(100, Math.min(1000, Math.abs(z) * 1000));
            
            coordinates.push({ x, y, z, laserPower, feedRate });
        }
        
        return coordinates;
    }

    async etchCrystalMemory(memoryId, coordinates) {
        return new Promise((resolve, reject) => {
            const stmt = this.db.prepare(
                `INSERT INTO crystal_etchings (memory_id, x_coord, y_coord, z_coord, laser_power, feed_rate, etching_data) 
                 VALUES (?, ?, ?, ?, ?, ?, ?)`
            );

            coordinates.forEach(coord => {
                stmt.run([
                    memoryId,
                    coord.x,
                    coord.y, 
                    coord.z,
                    coord.laserPower,
                    coord.feedRate,
                    JSON.stringify(coord)
                ]);
            });

            stmt.finalize((err) => {
                if (err) reject(err);
                else resolve();
            });
        });
    }

    // Generate G-code for CNC crystal etching
    generateGCode(memoryId) {
        return new Promise((resolve, reject) => {
            this.db.all(
                `SELECT * FROM crystal_etchings WHERE memory_id = ? ORDER BY id`,
                [memoryId],
                (err, rows) => {
                    if (err) {
                        reject(err);
                        return;
                    }

                    let gcode = '; G-code for Crystalline Memory Etching\n';
                    gcode += '; Generated by RENT A HAL MIM System\n';
                    gcode += 'G21 ; Set units to millimeters\n';
                    gcode += 'G90 ; Absolute positioning\n';
                    gcode += 'G0 X0 Y0 Z0 ; Move to origin\n';
                    gcode += 'M106 P0 ; Laser off\n\n';

                    rows.forEach((row, index) => {
                        gcode += `; Memory point ${index + 1}\n`;
                        gcode += `G1 X${row.x_coord.toFixed(3)} Y${row.y_coord.toFixed(3)} Z${row.z_coord.toFixed(3)} F${row.feed_rate}\n`;
                        gcode += `M106 P${row.laser_power} ; Laser power ${row.laser_power}\n`;
                        gcode += 'G4 P0.1 ; Dwell 0.1 seconds\n';
                        gcode += 'M106 P0 ; Laser off\n\n';
                    });

                    gcode += '; Return to origin\n';
                    gcode += 'G0 X0 Y0 Z0\n';
                    gcode += 'M30 ; End program\n';

                    resolve(gcode);
                }
            );
        });
    }

    // Recall memories based on resonance matching
    async recallMemories(currentState, resonanceThreshold = 0.8) {
        return new Promise((resolve, reject) => {
            const currentSignature = this.calculateResonanceSignature(currentState);
            
            this.db.all(
                `SELECT * FROM waveform_memory ORDER BY timestamp DESC LIMIT 100`,
                (err, rows) => {
                    if (err) {
                        reject(err);
                        return;
                    }

                    const matches = rows.filter(row => {
                        const similarity = this.calculateResonanceSimilarity(
                            currentSignature, 
                            row.resonance_signature
                        );
                        return similarity >= resonanceThreshold;
                    }).sort((a, b) => {
                        const simA = this.calculateResonanceSimilarity(currentSignature, a.resonance_signature);
                        const simB = this.calculateResonanceSimilarity(currentSignature, b.resonance_signature);
                        return simB - simA;
                    });

                    resolve(matches);
                }
            );
        });
    }

    calculateResonanceSimilarity(sig1, sig2) {
        if (sig1.length !== sig2.length) return 0;
        
        let matches = 0;
        for (let i = 0; i < sig1.length; i++) {
            if (sig1[i] === sig2[i]) matches++;
        }
        
        return matches / sig1.length;
    }
}

class ThreeMindsProcessor extends EventEmitter {
    constructor(memorySystem) {
        super();
        this.memorySystem = memorySystem;
        this.currentMind = new Map(); // Present processing
        this.pastMind = new Map();    // Memory/experience
        this.comparativeMind = new Map(); // Analysis/prediction
    }

    // Process through all three minds
    async processThrough(intent, sensorData, currentState) {
        const results = {
            current: await this.processCurrentMind(intent, sensorData, currentState),
            past: await this.processPastMind(intent, currentState),
            comparative: await this.processComparativeMind(intent, currentState)
        };

        // Synthesize decision from all three minds
        const synthesis = this.synthesizeDecision(results);
        
        // Store the processing result
        await this.memorySystem.storeWaveformMemory(
            intent.name,
            [results.current, results.past, results.comparative],
            { sensorData: Array.from(sensorData.entries()), currentState }
        );

        return synthesis;
    }

    async processCurrentMind(intent, sensorData, currentState) {
        // Real-time processing of immediate sensory input
        const currentWeight = intent.weight;
        const distance = intent.calculateDistance(currentState);
        
        // Calculate immediate response strength
        const immediateResponse = currentWeight * Math.exp(-distance);
        
        this.currentMind.set(intent.name, {
            weight: currentWeight,
            distance: distance,
            response: immediateResponse,
            timestamp: Date.now()
        });

        return immediateResponse;
    }

    async processPastMind(intent, currentState) {
        // Recall similar past experiences
        const memories = await this.memorySystem.recallMemories(
            Object.values(currentState), 
            0.7
        );

        let pastInfluence = 0;
        let memoryCount = 0;

        for (const memory of memories.slice(0, 5)) { // Top 5 matches
            try {
                const memoryState = JSON.parse(memory.state_vector);
                const timeDecay = Math.exp(-(Date.now() - memory.timestamp) / 86400000); // 24h decay
                pastInfluence += memoryState[0] * timeDecay; // Use first component
                memoryCount++;
            } catch (e) {
                console.warn('Error parsing memory state:', e);
            }
        }

        const pastResponse = memoryCount > 0 ? pastInfluence / memoryCount : 0;
        
        this.pastMind.set(intent.name, {
            memoryCount: memoryCount,
            influence: pastInfluence,
            response: pastResponse,
            timestamp: Date.now()
        });

        return pastResponse;
    }

    async processComparativeMind(intent, currentState) {
        // Analyze and predict based on current vs past
        const currentData = this.currentMind.get(intent.name);
        const pastData = this.pastMind.get(intent.name);

        if (!currentData || !pastData) {
            return intent.weight * 0.5; // Default response
        }

        // Compare current situation with past patterns
        const currentStrength = currentData.response;
        const pastStrength = pastData.response;
        
        // Predict future value based on trend
        const trend = currentStrength - pastStrength;
        const predictedValue = currentStrength + (trend * 0.5);
        
        // Confidence based on memory quality
        const confidence = Math.min(1.0, pastData.memoryCount / 3.0);
        const comparativeResponse = predictedValue * confidence;

        this.comparativeMind.set(intent.name, {
            trend: trend,
            prediction: predictedValue,
            confidence: confidence,
            response: comparativeResponse,
            timestamp: Date.now()
        });

        return comparativeResponse;
    }

    synthesizeDecision(results) {
        // Weight the three minds' inputs
        const currentWeight = 0.5;  // Present is most important
        const pastWeight = 0.3;     // Past experience significant
        const comparativeWeight = 0.2; // Analysis provides insight

        const synthesized = (
            results.current * currentWeight +
            results.past * pastWeight +
            results.comparative * comparativeWeight
        );

        return {
            decision: synthesized,
            breakdown: results,
            confidence: Math.min(1.0, synthesized / 1000)
        };
    }
}

class MasterIntentMatrix extends EventEmitter {
    constructor(cognitiveTemperature = 1.0) {
        super();
        this.cognitiveTemperature = cognitiveTemperature;
        this.realityMembrane = new RealityMembrane(0.5, cognitiveTemperature);
        this.memorySystem = new CrystallineMemorySystem();
        this.threeMinds = new ThreeMindsProcessor(this.memorySystem);
        this.intents = new Map();
        this.lastState = null;
        this.dt = 0.01;
        
        // Core RENT A HAL intents
        this.initializeCoreIntents();
        
        // Router state
        this.activeConnections = new Map();
        this.queryQueue = [];
        this.processingQuery = false;
    }

    initializeCoreIntents() {
        // Chat intent
        const chatIntent = new Intent('chat', 100000, { 
            queryType: 'chat', 
            urgency: 0.5, 
            complexity: 0.3 
        }, 0.3);
        chatIntent.sensors.set('text_input', 0.1);
        this.addIntent(chatIntent);

        // Vision intent  
        const visionIntent = new Intent('vision', 80000, {
            queryType: 'vision',
            imagePresent: 1.0,
            urgency: 0.7
        }, 0.5);
        visionIntent.sensors.set('image_upload', 0.8);
        this.addIntent(visionIntent);

        // Speech intent
        const speechIntent = new Intent('speech', 90000, {
            queryType: 'speech', 
            audioLevel: 0.8,
            wakeWord: 1.0
        }, 0.4);
        speechIntent.sensors.set('audio_input', 0.6);
        speechIntent.sensors.set('wake_word', 0.9);
        this.addIntent(speechIntent);

        // System monitoring intent
        const monitorIntent = new Intent('monitor', 50000, {
            systemHealth: 1.0,
            loadAverage: 0.3
        }, 0.2);
        monitorIntent.sensors.set('cpu_usage', 0.7);
        monitorIntent.sensors.set('memory_usage', 0.8);
        this.addIntent(monitorIntent);

        // Gmail intent
        const gmailIntent = new Intent('gmail', 70000, {
            emailCheck: 1.0,
            unreadCount: 0.5
        }, 0.3);
        gmailIntent.sensors.set('email_request', 0.8);
        this.addIntent(gmailIntent);
    }

    addIntent(intent) {
        this.intents.set(intent.name, intent);
        this.realityMembrane.addIntent(intent);
    }

    // Master differential equation for intent weight evolution
    calculateWeightDerivative(intent, sensorData, globalState) {
        const W = intent.weight;
        const S = this.processSensoryInput(intent, sensorData);
        const C = this.calculateInhibition(intent, globalState);
        const T = this.cognitiveTemperature;
        const distance = intent.currentDistance;

        // Core differential equation: dW/dt = S(1-W/Wmax)e^(-αD) - CW - λW + T√W N(0,1)
        const dW = (
            S * (1000000 - W) * Math.exp(-this.realityMembrane.decayRate * distance) -
            C * W -
            0.1 * W +
            T * (Math.random() - 0.5) * Math.sqrt(W)
        );

        return dW;
    }

    processSensoryInput(intent, sensorData) {
        let totalInput = 0;
        let sensorCount = 0;

        for (const [sensorId, threshold] of intent.sensors) {
            const value = sensorData.get(sensorId) || 0;
            if (value > threshold) {
                totalInput += value / threshold;
                sensorCount++;
            }
        }

        return sensorCount > 0 ? totalInput / sensorCount : 0;
    }

    calculateInhibition(intent, globalState) {
        let totalInhibition = 0;

        // Competitive inhibition from other active intents
        for (const [name, otherIntent] of this.intents) {
            if (name !== intent.name && otherIntent.activated) {
                const inhibitionStrength = otherIntent.weight / 1000000;
                totalInhibition += inhibitionStrength * 0.1;
            }
        }

        return Math.min(1, totalInhibition);
    }

    // Main routing decision method - replaces traditional routing
    async routeQuery(queryData, websocket, user) {
        const sensorData = new Map();
        const currentState = {};

        // Extract sensor data from query
        if (queryData.prompt) {
            sensorData.set('text_input', queryData.prompt.length / 100);
            currentState.queryType = 'chat';
        }

        if (queryData.image) {
            sensorData.set('image_upload', 1.0);
            currentState.queryType = 'vision';
            currentState.imagePresent = 1.0;
        }

        if (queryData.type === 'speech_to_text') {
            sensorData.set('audio_input', 0.8);
            sensorData.set('wake_word', 0.9);
            currentState.queryType = 'speech';
        }

        if (queryData.type === 'gmail_summary') {
            sensorData.set('email_request', 1.0);
            currentState.queryType = 'gmail';
        }

        // Set urgency and complexity
        currentState.urgency = queryData.urgent ? 1.0 : 0.5;
        currentState.complexity = this.estimateComplexity(queryData);

        // Update system using differential equations
        await this.updateSystem(sensorData, currentState, this.dt);

        // Find dominant intent
        const dominantIntent = await this.getDominantIntent(currentState);
        
        if (!dominantIntent) {
            throw new Error('No suitable intent found for query');
        }

        // Process through three minds
        const decision = await this.threeMinds.processThrough(
            dominantIntent, 
            sensorData, 
            currentState
        );

        // Route based on intent decision
        return await this.executeIntent(dominantIntent, queryData, websocket, user, decision);
    }

    async updateSystem(sensorData, globalState, deltaTime) {
        // Store current state
        this.lastState = {
            weights: new Map(Array.from(this.intents.entries()).map(([k, v]) => [k, v.weight])),
            sensorData: new Map(sensorData),  
            globalState: {...globalState}
        };

        // Update each intent using RK4 integration
        for (const intent of this.intents.values()) {
            const k1 = this.calculateWeightDerivative(intent, sensorData, globalState);
            const k2 = this.calculateWeightDerivative(intent, sensorData, globalState);
            const k3 = this.calculateWeightDerivative(intent, sensorData, globalState);
            const k4 = this.calculateWeightDerivative(intent, sensorData, globalState);

            // RK4 integration
            const dW = (k1 + 2*k2 + 2*k3 + k4) / 6;
            intent.weight = Math.max(0, Math.min(1000000, intent.weight + dW * deltaTime));

            // Check activation
            if (intent.shouldActivate(sensorData, globalState)) {
                intent.activated = true;
                intent.lastActivation = Date.now();
                intent.activationCount++;
            }
        }

        // Update reality membrane
        this.realityMembrane.updateSystem(deltaTime);
    }

    async getDominantIntent(currentState) {
        let maxPotential = -Infinity;
        let dominantIntent = null;

        for (const intent of this.intents.values()) {
            if (intent.activated) {
                const potential = this.realityMembrane.calculateIntentPotential(intent.name, currentState);
                if (potential > maxPotential) {
                    maxPotential = potential;  
                    dominantIntent = intent;
                }
            }
        }

        return dominantIntent;
    }

    estimateComplexity(queryData) {
        let complexity = 0.1;
        
        if (queryData.prompt) {
            complexity += Math.min(0.8, queryData.prompt.length / 1000);
        }
        
        if (queryData.image) {
            complexity += 0.6;
        }
        
        if (queryData.type === 'gmail_summary') {
            complexity += 0.4;
        }

        return Math.min(1.0, complexity);
    }

    async executeIntent(intent, queryData, websocket, user, decision) {
        this.emit('intent_execution', {
            intent: intent.name,
            decision: decision,
            user: user.guid,
            timestamp: Date.now()
        });

        // Route to appropriate handler based on intent
        switch (intent.name) {
            case 'chat':
                return await this.handleChatIntent(queryData, websocket, user, decision);
                
            case 'vision':
                return await this.handleVisionIntent(queryData, websocket, user, decision);
                
            case 'speech':
                return await this.handleSpeechIntent(queryData, websocket, user, decision);
                
            case 'gmail':
                return await this.handleGmailIntent(queryData, websocket, user, decision);
                
            case 'monitor':
                return await this.handleMonitorIntent(queryData, websocket, user, decision);
                
            default:
                throw new Error(`Unknown intent: ${intent.name}`);
        }
    }

    // Intent handlers - these replace the traditional route handlers
    async handleChatIntent(queryData, websocket, user, decision) {
        // Traditional chat processing with intent-driven prioritization
        const priority = decision.confidence > 0.8 ? 'high' : 'normal';
        
        await websocket.send(JSON.stringify({
            type: 'intent_activated',
            intent: 'chat',
            priority: priority,
            confidence: decision.confidence
        }));

        // Add to queue with intent priority
        return {
            type: 'chat',
            priority: priority,
            data: queryData,
            intent_decision: decision
        };
    }

    async handleVisionIntent(queryData, websocket, user, decision) {
        const priority = decision.confidence > 0.7 ? 'high' : 'normal';
        
        await websocket.send(JSON.stringify({
            type: 'intent_activated', 
            intent: 'vision',
            priority: priority,
            confidence: decision.confidence
        }));

        return {
            type: 'vision',
            priority: priority,
            data: queryData,  
            intent_decision: decision
        };
    }

    async handleSpeechIntent(queryData, websocket, user, decision) {
        // Speech has inherent urgency
        const priority = 'high';
        
        await websocket.send(JSON.stringify({
            type: 'intent_activated',
            intent: 'speech', 
            priority: priority,
            confidence: decision.confidence
        }));

        return {
            type: 'speech',
            priority: priority,
            data: queryData,
            intent_decision: decision
        };
    }

    async handleGmailIntent(queryData, websocket, user, decision) {
        const priority = decision.confidence > 0.6 ? 'high' : 'normal';
        
        await websocket.send(JSON.stringify({
            type: 'intent_activated',
            intent: 'gmail',
            priority: priority, 
            confidence: decision.confidence
        }));

        return {
            type: 'gmail',
            priority: priority,
            data: queryData,
            intent_decision: decision
        };
    }

    async handleMonitorIntent(queryData, websocket, user, decision) {
        // System monitoring is background priority
        const priority = 'low';
        
        return {
            type: 'monitor',
            priority: priority,
            data: queryData,
            intent_decision: decision
        };
    }

    // Generate G-code for crystalline memory etching
    async generateMemoryGCode(intentName) {
        const memories = await this.memorySystem.recallMemories([intentName], 0.5);
        
        if (memories.length === 0) {
            return null;
        }

        const memoryId = memories[0].id;
        return await this.memorySystem.generateGCode(memoryId);
    }

    // Get system status for monitoring
    getSystemStatus() {
        const status = {
            intents: {},
            membrane: {
                temperature: this.cognitiveTemperature,
                decayRate: this.realityMembrane.decayRate,
                activeIntents: 0
            },
            timestamp: Date.now()
        };

        for (const [name, intent] of this.intents) {
            status.intents[name] = {
                weight: intent.weight,
                activated: intent.activated,
                lastActivation: intent.lastActivation,
                activationCount: intent.activationCount
            };
            
            if (intent.activated) {
                status.membrane.activeIntents++;
            }
        }

        return status;
    }

    // Cleanup method
    async destroy() {
        if (this.memorySystem && this.memorySystem.db) {
            this.memorySystem.db.close();
        }
        this.removeAllListeners();
    }
}

// Export for use as drop-in replacement
export { MasterIntentMatrix, Intent, RealityMembrane, CrystallineMemorySystem, ThreeMindsProcessor };

// Usage example for integration:
/*
// Replace traditional router with MIM
const mim = new MasterIntentMatrix(1.0);

// Handle incoming queries  
app.post('/query', async (req, res) => {
    try {
        const result = await mim.routeQuery(req.body, websocket, user);
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Generate G-code for memory etching
app.get('/memory/gcode/:intent', async (req, res) => {
    const gcode = await mim.generateMemoryGCode(req.params.intent);
    res.set('Content-Type', 'text/plain');
    res.send(gcode);
});

// System status endpoint
app.get('/mim/status', (req, res) => {
    res.json(mim.getSystemStatus());
});
*/