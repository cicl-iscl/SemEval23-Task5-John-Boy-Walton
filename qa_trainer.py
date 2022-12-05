from transformers import Trainer


class QATrainer(Trainer):

    def __init__(self, post_process_function, post_process_kwargs, *args, **kwargs):
        super(QATrainer, self).__init__(*args, **kwargs)
        self.post_process_function = post_process_function
        self.post_process_kwargs = post_process_kwargs

    def evaluate(self, eval_dataset=None, ignore_keys=None):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # temporarily disable metric computation, we will do it in the loop here
        # because evaluation_loop() wants compute metrics right away
        # if self.compute_metrics is not None but ours requires additional args
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        # eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        # try:
        output = self.evaluation_loop(
            eval_dataloader,
            description='Evaluation',
            # don't gather predictions if there are no metric
            # prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys
        )
        # finally:
        self.compute_metrics = compute_metrics

        # if (self.post_process_function is not None) and (self.compute_metrics is not None):
        eval_preds = self.post_process_function(eval_dataset, output.predictions, **self.post_process_kwargs)
        metrics = self.compute_metrics(eval_preds)

            # self.log(metrics)
        # else:
        #     metrics = {}

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        log_history = self.state.log_history # dict      
        if len(log_history):
            all_metrics = {**log_history[-1], **metrics}
            print(all_metrics)
            try:
                self.save_metrics("eval", all_metrics)
            except: pass
        return metrics