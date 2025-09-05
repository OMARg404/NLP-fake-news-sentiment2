import axios from 'axios';

const API_URL = 'http://127.0.0.1:8000';

export const predict = async(text) => {
    try {
        const resp = await axios.post(`${API_URL}/predict`, { text });
        return resp.data;
    } catch (error) {
        console.error("API error:", error);
        return null;
    }
};