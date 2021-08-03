import torch
import torch.nn.functional as F
from allennlp.training.metrics import Metric


class ExtractorMetric(Metric):
    def __init__(self):
        self.i_g_tp = self.i_p_tp = 0.
        self.c_tp = 0.
        self.i_fp = self.c_fp = 0.
        self.i_fn = self.c_fn = 0.

    def __call__(self, pred_spans, gold_spans):
        assert len(pred_spans) == len(gold_spans)
        if isinstance(gold_spans, torch.Tensor):
            gold_spans = gold_spans.cpu().numpy().tolist()
        batch_size = len(pred_spans)

        for idx in range(batch_size):
            sentence_pred_span = pred_spans[idx]
            sentence_gold_span = gold_spans[idx]

            gold_span_i_flag = [0 for i in range(len(sentence_gold_span))]
            gold_span_c_flag = [0 for i in range(len(sentence_gold_span))]

            pred_span_i_flag = [0 for i in range(len(sentence_pred_span))]
            pred_span_c_flag = [0 for i in range(len(sentence_pred_span))]

            pred_span_num = len(sentence_pred_span)
            gold_span_num = 0

            '''
            i_pred_tot_num = 0
            flag = [True for i in range(len(sentence_pred_span))]
            for i in range(len(sentence_pred_span)):
                pred_span_num += 1
                if not flag[i]:
                    continue
                i_pred_tot_num += 1
                for j in range(i+1, len(sentence_pred_span)):
                    if not flag[j]:
                        continue
                    if (sentence_pred_span[i][0] == sentence_pred_span[j][0]) and (sentence_pred_span[i][1] == sentence_pred_span[j][1]):
                        flag[j] = False
            '''

            for i, g_span in enumerate(sentence_gold_span):
                if g_span[0] == -1:
                    break
                gold_span_num += 1
                for j, p_span in enumerate(sentence_pred_span):
                    if (g_span[0] == p_span[0]) and (g_span[1] == p_span[1]):
                        gold_span_i_flag[i] = 1
                        pred_span_i_flag[j] = 1

                    if (g_span[0] == p_span[0]) and (g_span[1] == p_span[1]) and (g_span[2] == p_span[2]):
                        gold_span_c_flag[i] = 1
                        pred_span_c_flag[j] = 1

            i_g_tp = sum(gold_span_i_flag)
            i_p_tp = sum(pred_span_i_flag)

            i_fn = gold_span_num - i_g_tp
            i_fp = pred_span_num - i_p_tp

            c_tp = sum(gold_span_c_flag)
            c_fn = gold_span_num - c_tp
            c_fp = pred_span_num - c_tp

            self.i_g_tp += i_g_tp
            self.i_p_tp += i_p_tp
            self.i_fp += i_fp
            self.i_fn += i_fn
            self.c_tp += c_tp
            self.c_fp += c_fp
            self.c_fn += c_fn

    def _get_p_r_f(self, tp, fp, fn):
        p = float(tp) / float(tp + fp + 1e-13)
        r = float(tp) / float(tp + fn + 1e-13)
        f = 2. * ((p * r) / (p + r + 1e-13))
        return p, r, f

    def _get_i_p_r_f(self, g_tp, p_tp, fp, fn):
        p = float(p_tp) / float(p_tp + fp + 1e-13)
        r = float(g_tp) / float(g_tp + fn + 1e-13)
        f = 2. * ((p * r) / (p + r + 1e-13))
        return p, r, f

    def get_metric(self, reset=False):
        i_p, i_r, i_f = self._get_i_p_r_f(self.i_g_tp, self.i_p_tp, self.i_fp, self.i_fn)
        c_p, c_r, c_f = self._get_p_r_f(self.c_tp, self.c_fp, self.c_fn)
        if reset:
            self.reset()
        return i_p, i_r, i_f, c_p, c_r, c_f

    def reset(self):
        self.i_g_tp = self.i_p_tp = 0.
        self.c_tp = 0.
        self.i_fp = self.c_fp = 0.
        self.i_fn = self.c_fn = 0.


class TriggerMetric(Metric):
    def __init__(self):
        super().__init__()
        self.metric = ExtractorMetric()

    def __call__(self, logits, triggers):
        pred_span = self.get_span(logits)
        self.metric(pred_span, triggers)

    def get_metric(self, reset):
        i_p, i_r, i_f, c_p, c_r, c_f = self.metric.get_metric()
        if reset:
            self.metric.reset()
        return {"t_i_p": i_p, "t_i_r": i_r, "t_i_f": i_f,
                "t_c_p": c_p, "t_c_r": c_r, "t_c_f": c_f}

    def _get_single_classifier_span(self, logits, label):
        batch_size = logits.size(0)
        prediciton = torch.argmax(logits, -1)           # B x L

        label_span = [[] for i in range(batch_size)]
        for i, sequence in enumerate(prediciton):
            spans = []
            idx = 0
            while idx < sequence.size(0):
                if sequence[idx] == 1 and (idx == 0 or sequence[idx-1] == 0):
                    start = idx
                    while idx+1 < sequence.size(0) and sequence[idx+1] == 1:
                        idx += 1
                    end = idx
                    spans.append([start, end, label])
                idx += 1
            label_span[i] = spans
        return label_span

    def get_span(self, logits):
        batch_size = logits[0].size(0)
        spans = [[] for i in range(batch_size)]
        for i, logit in enumerate(logits):
            label_span = self._get_single_classifier_span(logit, i)
            for b in range(batch_size):
                spans[b].extend(label_span[b])
        return spans


class ArgumentMetric(Metric):
    def __init__(self, event_roles, threshold):
        super().__init__()
        self.metric = ExtractorMetric()
        self.event_roles = event_roles
        self.threshold = threshold

    def __call__(self, start_logits, end_logits, event_type, roles):
        filtered_span = self.get_span(start_logits, end_logits, event_type)
        self.metric(filtered_span, roles)

    def get_span(self, start_logits, end_logits, event_type):
        # self._get_single_event_span(start_logits[0], end_logits[0], 0)
        assert len(start_logits) == len(end_logits)
        batch_size = event_type.size(0)
        pred_span = [[] for i in range(batch_size)]
        for idxt in range(len(start_logits)):
            tagger_span = self._get_single_event_span(start_logits[idxt], end_logits[idxt], idxt, event_type)
            for idxb in range(batch_size):
                pred_span[idxb].extend(tagger_span[idxb])
        return pred_span

    def _get_single_event_span(self, start_logit, end_logit, label, event_type):
        batch_size, seq_len, _ = start_logit.size()

        start_prob = F.softmax(start_logit, dim=-1)
        end_prob = F.softmax(end_logit, dim=-1)

        start_label = start_prob[:, :, 1] > self.threshold
        end_label = end_prob[:, :, 1] > self.threshold

        single_event_spans = [[] for i in range(batch_size)]
        for idb in range(batch_size):
            sentence_span = []
            state = 'none'
            s_pos, s_prob = 0, 0.
            e_pos, e_prob = 0, 0.

            et_id = int(event_type[idb])
            if label not in self.event_roles[et_id]:
                continue

            for idw in range(seq_len):
                if state == 'none':
                    if int(start_label[idb, idw]) == 1:
                        state = 'start'
                        s_pos = idw
                        s_prob = float(start_prob[idb, idw, 1])
                if state == 'start':
                    if (int(start_label[idb, idw]) == 1) and (s_pos < idw):
                        new_s_prob = float(start_prob[idb, idw, 1])
                        if new_s_prob > s_prob:
                            s_pos = idw
                            s_prob = new_s_prob
                    if int(end_label[idb, idw]) == 1:
                        state = 'end'
                        e_pos = idw
                        e_prob = float(end_prob[idb, idw, 1])
                if state == 'end':
                    if e_pos < idw:
                        if (int(start_label[idb, idw]) == 1) and (int(end_label[idb, idw]) == 1):
                            sentence_span.append([s_pos, e_pos, label, et_id])
                            s_pos = idw
                            s_prob = float(start_prob[idb, idw, 1])
                            e_pos = idw
                            e_prob = float(end_prob[idb, idw, 1])

                        elif int(start_label[idb, idw]) == 1:
                            sentence_span.append([s_pos, e_pos, label, et_id])
                            state = 'start'
                            s_pos = idw
                            s_prob = float(start_prob[idb, idw, 1])

                        elif int(end_label[idb, idw]) == 1:
                            new_e_prob = float(end_prob[idb, idw, 1])
                            if new_e_prob > e_prob:
                                e_pos = idw
                                e_prob = new_e_prob
                if (idw == seq_len - 1) and (state == 'end'):
                    sentence_span.append([s_pos, e_pos, label, et_id])
            single_event_spans[idb] = sentence_span
        return single_event_spans

    def get_metric(self, reset):
        i_p, i_r, i_f, c_p, c_r, c_f = self.metric.get_metric()
        if reset:
            self.metric.reset()
        return {"r_i_p": i_p, "r_i_r": i_r, "r_i_f": i_f,
                "r_c_p": c_p, "r_c_r": c_r, "r_c_f": c_f}
