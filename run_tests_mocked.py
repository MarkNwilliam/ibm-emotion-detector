import unittest
from unittest.mock import patch, Mock
from EmotionDetection.emotion_detection import emotion_detector

class TestEmotionDetectionMocked(unittest.TestCase):
    @patch('requests.post')
    def test_emotion_detector(self, mock_post):
        # Define mock responses for different inputs
        def side_effect(url, json, headers):
            text = json['raw_document']['text']
            mock_resp = Mock()
            mock_resp.status_code = 200
            if 'glad' in text:
                emotions = {"anger": 0.0, "disgust": 0.0, "fear": 0.0, "joy": 0.9, "sadness": 0.1}
            elif 'mad' in text:
                emotions = {"anger": 0.9, "disgust": 0.0, "fear": 0.0, "joy": 0.0, "sadness": 0.1}
            elif 'disgusted' in text:
                emotions = {"anger": 0.0, "disgust": 0.9, "fear": 0.0, "joy": 0.0, "sadness": 0.1}
            elif 'afraid' in text:
                emotions = {"anger": 0.0, "disgust": 0.0, "fear": 0.9, "joy": 0.0, "sadness": 0.1}
            elif 'sad' in text:
                emotions = {"anger": 0.0, "disgust": 0.0, "fear": 0.0, "joy": 0.0, "sadness": 0.9}
            else:
                emotions = {"anger": 0.1, "disgust": 0.1, "fear": 0.1, "joy": 0.1, "sadness": 0.1}
            
            mock_resp.text = '{"emotionPredictions": [{"emotion": ' + str(emotions).replace("'", '"') + '}]}'
            return mock_resp

        mock_post.side_effect = side_effect
        
        # Test case for joy
        result_1 = emotion_detector('I am glad this happened')
        self.assertEqual(result_1['dominant_emotion'], 'joy')
        
        # Test case for anger
        result_2 = emotion_detector('I am really mad about this')
        self.assertEqual(result_2['dominant_emotion'], 'anger')
        
        # Test case for disgust
        result_3 = emotion_detector('I feel disgusted just hearing about this')
        self.assertEqual(result_3['dominant_emotion'], 'disgust')
        
        # Test case for fear
        result_4 = emotion_detector('I am really afraid that this will happen')
        self.assertEqual(result_4['dominant_emotion'], 'fear')
        
        # Test case for sadness
        result_5 = emotion_detector('I am so sad about this')
        self.assertEqual(result_5['dominant_emotion'], 'sadness')

if __name__ == '__main__':
    unittest.main()
