"""
Unit Tests for Homs Real Estate Prediction Flask Application
============================================================
To run tests: pytest test_app.py  OR  python -m unittest test_app.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import json
from app import app, bundle


class RealEstateAppTestCase(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

    def test_home_route(self):
        """1. اختبار صحة استجابة الصفحة الرئيسية"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_api_status_route(self):
        """2. اختبار نقطة نهاية فحص حالة الخدمة"""
        response = self.client.get('/api/status')
        if bundle is not None:
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data['status'], 'ready')
        else:
            self.assertEqual(response.status_code, 503)

    def test_api_locations_route(self):
        """3. اختبار نقطة نهاية جلب قائمة المناطق المتاحة"""
        response = self.client.get('/api/locations')
        if bundle is not None:
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertTrue(data.get('success'))
            self.assertIsInstance(data.get('locations'), list)
        else:
            self.assertEqual(response.status_code, 503)

    def test_valid_form_predict_route(self):
        """4. اختبار مسار التوقع بالنموذج عبر الاستمارة (Form Post)"""
        if bundle is None:
            self.skipTest("Model is not loaded.")

        valid_payload = {
            "location": list(bundle["target_enc_map"].keys())[0] if bundle.get("target_enc_map") else "الحمراء",
            "total_area": 120,
            "net_area": 100,
            "rooms_count": 3,
            "bathrooms_count": 1,
            "halls_count": 1,
            "floor_num": 2,
            "building_age": 5,
            "heating_type": "تدفئة ديزل",
            "house_condition": "مسكون من صاحبه",
            "legal_status": "طابو أخضر/نظامي",
            "has_elevator": "1",
            "is_furnished": "0",
        }
        response = self.client.post('/predict', data=valid_payload, follow_redirects=True)
        self.assertEqual(response.status_code, 200)

    def test_api_predict_json_success(self):
        """5. اختبار API التوقع بإرسال JSON صحيح"""
        if bundle is None:
            self.skipTest("Model is not loaded.")

        json_data = {
            "location": list(bundle["target_enc_map"].keys())[0] if bundle.get("target_enc_map") else "الحمراء",
            "total_area": 150,
            "net_area": 130,
            "rooms_count": 4,
            "bathrooms_count": 2,
            "halls_count": 1,
            "floor_num": 3,
            "building_age": 10,
            "heating_type": "تدفئة مركزية",
            "house_condition": "فارغ",
            "legal_status": "طابو أخضر/نظامي",
            "near_hospital": 1,
            "has_elevator": 1,
        }
        response = self.client.post(
            '/api/predict',
            data=json.dumps(json_data),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        res_json = json.loads(response.data)
        self.assertTrue(res_json.get('success'))
        self.assertIn('predicted_price_usd', res_json)

    def _base_json_payload(self, floor_num):
        """دالة مساعدة لبناء بيانات الاختبار للحدود"""
        return {
            "location": list(bundle["target_enc_map"].keys())[0] if bundle.get("target_enc_map") else "الحمراء",
            "total_area": 120,
            "net_area": 100,
            "rooms_count": 3,
            "bathrooms_count": 1,
            "halls_count": 1,
            "floor_num": floor_num,
            "building_age": 5,
            "heating_type": "تدفئة ديزل",
            "house_condition": "مسكون من صاحبه",
            "legal_status": "طابو أخضر/نظامي",
        }

    def test_floor_num_lower_boundary_accepted(self):
        """6. قبول الحد الأدنى لرقم الطابق (-3)"""
        if bundle is None:
            self.skipTest("Model is not loaded.")

        response = self.client.post(
            '/api/predict',
            data=json.dumps(self._base_json_payload(-3)),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(json.loads(response.data).get('success'))

    def test_floor_num_upper_boundary_accepted(self):
        """7. قبول الحد الأعلى لرقم الطابق (15)"""
        if bundle is None:
            self.skipTest("Model is not loaded.")

        response = self.client.post(
            '/api/predict',
            data=json.dumps(self._base_json_payload(15)),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(json.loads(response.data).get('success'))

    def test_floor_num_below_range_rejected(self):
        """8. رفض القيمة أقل من الحد الأدنى لطابق (-4)"""
        if bundle is None:
            self.skipTest("Model is not loaded.")

        response = self.client.post(
            '/api/predict',
            data=json.dumps(self._base_json_payload(-4)),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

    def test_floor_num_above_range_rejected(self):
        """9. رفض القيمة أعلى من الحد الأعلى لطابق (16)"""
        if bundle is None:
            self.skipTest("Model is not loaded.")

        response = self.client.post(
            '/api/predict',
            data=json.dumps(self._base_json_payload(16)),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

    def test_invalid_net_area_error_handling(self):
        """10. اختبار كشف الخطأ عند إدخال مساحة صافية أكبر من الإجمالية"""
        if bundle is None:
            self.skipTest("Model is not loaded.")

        invalid_data = {
            "location": "الحمراء",
            "total_area": 100,
            "net_area": 150,
            "rooms_count": 3,
            "heating_type": "لا يوجد",
            "house_condition": "فارغ",
            "legal_status": "طابو أخضر/نظامي",
        }
        response = self.client.post(
            '/api/predict',
            data=json.dumps(invalid_data),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()