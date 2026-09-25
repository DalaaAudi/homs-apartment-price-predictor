import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { htmlReport } from "https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js";
import { textSummary } from "https://jslib.k6.io/k6-summary/0.0.1/index.js";

export const options = {
  stages: [
    { duration: '10s', target: 5 },
    { duration: '20s', target: 15 },
    { duration: '10s', target: 0 },
  ],
  thresholds: {
    http_req_failed: ['rate<0.05'],
    http_req_duration: ['p(95)<2500'],
  },
};

const BASE_URL = 'http://127.0.0.1:5092';

export default function () {
  group('01. Flask ML Prediction API', function () {
    const apiPayload = JSON.stringify({
      location: "الحمراء",
      total_area: 120,
      net_area: 100,
      rooms_count: 3,
      bathrooms_count: 2,
      halls_count: 1,
      floor_num: 2,
      building_age: 5,
      heating_type: "تدفئة ديزل",
      house_condition: "مسكون من صاحبه",
      legal_status: "طابو أخضر/نظامي",
      near_hospital: 1,
      near_market: 1,
      has_elevator: 1,
      has_parking: 0,
      is_furnished: 0
    });

    const params = {
      headers: { 'Content-Type': 'application/json' },
    };

    let resApi = http.post(`${BASE_URL}/api/predict`, apiPayload, params);
    check(resApi, {
      'AI Predict Status 200': (r) => r.status === 200,
    });
  });

  sleep(1);
}

export function handleSummary(data) {
  return {
    "flask_summary.html": htmlReport(data),
    stdout: textSummary(data, { indent: " ", enableColors: true }),
  };
}