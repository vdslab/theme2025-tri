import { ApiService } from "./ApiService";

export class ClusterVisualizationService {
  static async getLogisticRegressionData(gamepk) {
    return ApiService.callGetApi(
      "api/clusterVisualization/logistic-regression-data",
      {
        gamepk: gamepk,
      },
    );
  }
}
