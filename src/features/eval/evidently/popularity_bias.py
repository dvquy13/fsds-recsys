from typing import Optional

import pandas as pd
from evidently.base_metric import InputData
from evidently.calculations.recommender_systems import get_prediciton_name
from evidently.metrics import PopularityBias
from evidently.metrics.recsys.popularity_bias import PopularityBiasResult
from evidently.pipeline.column_mapping import RecomType
from evidently.utils.visualizations import get_distribution_for_column


class CustomPopularityBias(PopularityBias):
    """Customize this metric report to fix incorrect measurement for reference dataset"""

    def calculate(self, data: InputData) -> PopularityBiasResult:
        train_result = self._train_stats.get_result()
        curr_user_interacted = train_result.current
        ref_user_interacted = train_result.reference
        prediction_name = get_prediciton_name(data)
        col_user_id = data.data_definition.get_user_id_column()
        col_item_id = data.data_definition.get_item_id_column()
        recommendations_type = data.column_mapping.recom_type
        if col_user_id is None or col_item_id is None or recommendations_type is None:
            raise ValueError(
                "user_id and item_id and recommendations_type should be specified"
            )
        user_id = col_user_id.column_name
        item_id = col_item_id.column_name

        current_data = data.current_data.copy()
        if recommendations_type == RecomType.SCORE:
            current_data[prediction_name] = current_data.groupby(user_id)[
                prediction_name
            ].transform("rank", ascending=False)

        current_apr, current_distr_data = self.get_apr(
            self.k,
            current_data,
            curr_user_interacted,
            self.normalize_arp,
            prediction_name,
            user_id,
            item_id,
        )
        curr_coverage = self.get_coverage(
            self.k, current_data, curr_user_interacted, prediction_name, item_id
        )

        curr_gini = self.get_gini(self.k, current_data, prediction_name, item_id)

        reference_apr: Optional[float] = None
        ref_coverage: Optional[float] = None
        ref_gini: Optional[float] = None
        reference_distr_data: Optional[pd.Series] = None
        if data.reference_data is not None:
            reference_data = data.reference_data.copy()
            if recommendations_type == RecomType.SCORE:
                reference_data[prediction_name] = reference_data.groupby(user_id)[
                    prediction_name
                ].transform("rank", ascending=False)
            if ref_user_interacted is None:
                ref_user_interacted = curr_user_interacted

            reference_apr, reference_distr_data = self.get_apr(
                self.k,
                reference_data,
                ref_user_interacted,
                self.normalize_arp,
                prediction_name,
                user_id,
                item_id,
            )

            # <Quy> Use get_coverage for consistent result and use self.k instead of all the data submitted
            # This issue can lead to incorrect reference coverage especially for popular recommendations
            ref_coverage = self.get_coverage(
                self.k, reference_data, ref_user_interacted, prediction_name, item_id
            )
            # ref_coverage = reference_data[item_id].nunique() / len(ref_user_interacted)
            # </Quy>

            ref_gini = self.get_gini(self.k, reference_data, prediction_name, item_id)
        current_distr, reference_distr = get_distribution_for_column(
            column_type="num",
            current=current_distr_data,
            reference=reference_distr_data,
        )

        return PopularityBiasResult(
            k=self.k,
            normalize_arp=self.normalize_arp,
            current_apr=current_apr,
            current_coverage=curr_coverage,
            current_gini=curr_gini,
            current_distr=current_distr,
            reference_apr=reference_apr,
            reference_coverage=ref_coverage,
            reference_distr=reference_distr,
            reference_gini=ref_gini,
        )
