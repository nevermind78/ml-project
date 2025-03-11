import unittest
from api.api import app
import numpy as np
from PIL import Image
import io

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_predict_endpoint(self):
        # Create a dummy image
        image = Image.new('L', (28, 28), color=0)
        image_byte_array = io.BytesIO()
        image.save(image_byte_array, format='PNG')
        image_byte_array.seek(0)

        # Send a POST request to the /predict endpoint
        response = self.client.post(
            '/predict',
            data={
                'file': (image_byte_array, 'test_image.png'),
                'model': 'Logistic Regression'
            }
        )

        # Check the response
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', response.json)

if __name__ == '__main__':
    unittest.main()