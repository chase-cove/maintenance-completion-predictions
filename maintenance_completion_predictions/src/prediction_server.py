from concurrent import futures
import logging

import grpc
import prediction_service_pb2
import prediction_service_pb2_grpc

from utils import make_prediction


class Predictions(prediction_service_pb2_grpc.PredictionServicer):
    def PredictMultiple(self, request, context):
        tasks_data = [
            {
                "dueDate": task.due_date,
                "estimatedHours": task.estimated_hours,
                "completedOn": "",
                "name": "",
                "currentLaborHours": task.current_labor_hours,
                "categoryName": "",
                "siteName": "",
                "createdAt": task.created_at,
                "isCompletedOnTime": False,
            }
            for task in request.tasks
        ]

        predictions = make_prediction(tasks_data)

        return prediction_service_pb2.PredictResponse(prediction_results=predictions)


def serve():
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    prediction_service_pb2_grpc.add_PredictionServicer_to_server(Predictions(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()
