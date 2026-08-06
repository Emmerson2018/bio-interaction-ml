import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from base_tool.visualization.base_visualizer import BaseVisualizer
from base_tool.utils.registry import VISUALIZATION_REGISTRY


@VISUALIZATION_REGISTRY.register()
class PerClassMetricsVisualizer(BaseVisualizer):
    def __init__(self, opt):
        super().__init__(opt)
        self.histories = {} 
        self.epochs = []

    def visualize(self, current_iter, payload):
        if 'val_probabilities' not in payload:
            return
            
        probs = payload['val_probabilities'].numpy()
        targets = payload['val_targets'].numpy()
        preds = np.argmax(probs, axis=1)
        classes = payload.get('classes', [str(i) for i in range(probs.shape[1])])
        epoch = payload.get('epoch', 0)
        
        if epoch not in self.epochs:
            self.epochs.append(epoch)
            
        report = classification_report(targets, preds, target_names=classes, output_dict=True, zero_division=0)
        
        for cls in classes:
            if cls not in self.histories:
                self.histories[cls] = {'f1': [], 'precision': [], 'recall': []}
            
            if cls in report:
                self.histories[cls]['f1'].append(report[cls]['f1-score'])
                self.histories[cls]['precision'].append(report[cls]['precision'])
                self.histories[cls]['recall'].append(report[cls]['recall'])
            else:
                self.histories[cls]['f1'].append(0)
                self.histories[cls]['precision'].append(0)
                self.histories[cls]['recall'].append(0)
                
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        metrics = ['f1', 'precision', 'recall']
        titles = ['F1-Score per Class', 'Precision per Class', 'Recall per Class']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            for cls in classes:
                ax.plot(self.epochs, self.histories[cls][metric], label=cls, marker='o')
            ax.set_title(titles[idx])
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.grid(True)
            if idx == 2: 
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
        plt.tight_layout()
        save_file = os.path.join(self.save_path, 'per_class_metrics.png')
        plt.savefig(save_file)
        plt.close()


@VISUALIZATION_REGISTRY.register()
class ConfidenceBoxplotVisualizer(BaseVisualizer):
    def __init__(self, opt):
        super().__init__(opt)
        
    def visualize(self, current_iter, payload):
        if 'val_probabilities' not in payload:
            return
            
        probs = payload['val_probabilities'].numpy()
        targets = payload['val_targets'].numpy()
        preds = np.argmax(probs, axis=1)
        classes = payload.get('classes', [str(i) for i in range(probs.shape[1])])
        
        data = []
        for i in range(len(preds)):
            pred_class = preds[i]
            conf = probs[i, pred_class]
            is_correct = (pred_class == targets[i])
            class_name = classes[pred_class] if pred_class < len(classes) else str(pred_class)
            
            data.append({
                'Class': class_name,
                'Confidence': conf,
                'Status': 'Correct' if is_correct else 'Incorrect'
            })
            
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 6))
        if not df.empty:
            sns.boxplot(x='Class', y='Confidence', hue='Status', data=df, palette={'Correct': 'g', 'Incorrect': 'r'})
        plt.title(f"Prediction Confidence by Class - Epoch {payload.get('epoch', 0)}")
        plt.ylim(0, 1.05)
        plt.grid(True, axis='y')
        
        save_file = os.path.join(self.save_path, 'confidence_boxplot.png')
        plt.savefig(save_file)
        plt.close()


@VISUALIZATION_REGISTRY.register()
class ConfusionMatrixVisualizer(BaseVisualizer):
    def __init__(self, opt):
        super().__init__(opt)
        
    def visualize(self, current_iter, payload):
        if 'val_probabilities' not in payload:
            return
            
        epoch = payload.get('epoch', 0)
        total_epochs = payload.get('total_epochs', 0)
        
        if epoch < total_epochs - 1 and total_epochs > 0:
            return
            
        probs = payload['val_probabilities'].numpy()
        targets = payload['val_targets'].numpy()
        preds = np.argmax(probs, axis=1)
        classes = payload.get('classes', [str(i) for i in range(probs.shape[1])])
        
        cm = confusion_matrix(targets, preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        save_file = os.path.join(self.save_path, 'confusion_matrix.png')
        plt.savefig(save_file)
        plt.close()
