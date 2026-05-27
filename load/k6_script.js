import http from "k6/http";
import { check, sleep } from "k6";

const baseUrl = __ENV.BASE_URL || "http://localhost:8000";

export const options = {
  scenarios: {
    bursty_cpu: {
      executor: "ramping-arrival-rate",
      startRate: 1,
      timeUnit: "1s",
      preAllocatedVUs: 30,
      maxVUs: 100,
      stages: [
        { duration: "45s", target: 2 },
        { duration: "60s", target: 20 },
        { duration: "45s", target: 6 },
        { duration: "60s", target: 35 },
        { duration: "45s", target: 4 },
        { duration: "60s", target: 24 },
        { duration: "45s", target: 1 },
      ],
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.10"],
    http_req_duration: ["p(95)<2000"],
  },
};

export default function () {
  const work = [20, 35, 50, 80, 120][Math.floor(Math.random() * 5)];
  const response = http.get(`${baseUrl}/cpu?work=${work}`, {
    tags: { endpoint: "cpu" },
    timeout: "5s",
  });
  check(response, {
    "status is 200": (r) => r.status === 200,
  });
  sleep(Math.random() * 0.2);
}
