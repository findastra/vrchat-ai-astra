import sys
import os
import sqlite3
import json
import pandas as pd
import networkx as nx
import numpy as np
import pyvista as pv
import ast
import gc
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QFrame, QSlider, QCheckBox,
    QMessageBox, QGroupBox
)
from PySide6.QtCore import Qt
from pyvistaqt import QtInteractor

# --- Constants for File Names ---
DB_FILE = 'mai_phoenix_brain.db'
SEMANTIC_CLUSTERS_FILE = 'mai_semantic_clusters.json'
ATTENTION_FILE = 'mai_attention_weights.json'

class BrainDataLoader:
    """Loads and processes all brain-related files into a unified graph structure."""

    def __init__(self, data_path):
        self.path = data_path
        self.graph = nx.DiGraph()
        self.clusters = {}
        self.attention_weights = {}
        self.associations_df = pd.DataFrame()
        self.status_messages = []

    def load_data(self):
        """Loads all data sources and returns True on success, False on failure."""
        try:
            self._load_semantic_clusters()
            self._load_attention_weights()
            self._load_word_associations()
            self._populate_graph()

            if self.graph.number_of_nodes() == 0:
                self.status_messages.append("Warning: Files were loaded, but no valid data was found to visualize.")
                self.status_messages.append("This usually means the AI needs more training to form stronger connections and clusters.")
            
            return True
        except FileNotFoundError as e:
            self.status_messages.append(f"Error: Required file not found.")
            self.status_messages.append(str(e))
            return False
        except Exception as e:
            self.status_messages.append(f"An unexpected error occurred during loading: {e}")
            return False

    def _load_semantic_clusters(self):
        cluster_file = os.path.join(self.path, SEMANTIC_CLUSTERS_FILE)
        if not os.path.exists(cluster_file):
            raise FileNotFoundError(f"'{SEMANTIC_CLUSTERS_FILE}' was not found in the selected directory.")
        with open(cluster_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.clusters = {k: v for k, v in data.get('clusters', {}).items() if len(v) > 1}
        if not self.clusters:
            self.status_messages.append(f"Loaded '{SEMANTIC_CLUSTERS_FILE}', but found 0 valid clusters (clusters must have >1 word).")
        else:
            self.status_messages.append(f"Loaded {len(self.clusters)} clusters from '{SEMANTIC_CLUSTERS_FILE}'.")

    def _load_attention_weights(self):
        attention_file = os.path.join(self.path, ATTENTION_FILE)
        if os.path.exists(attention_file):
            with open(attention_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.attention_weights = data.get('focus_weights', {})
            self.status_messages.append(f"Loaded {len(self.attention_weights)} attention weights.")
        else:
            self.status_messages.append(f"Note: '{ATTENTION_FILE}' not found, attention data will be default.")

    def _load_word_associations(self):
        db_path = os.path.join(self.path, DB_FILE)
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"'{DB_FILE}' was not found in the selected directory.")
        con = None
        try:
            con = sqlite3.connect(db_path)
            query = "SELECT source_word, next_word, priority, success_rate FROM word_associations WHERE priority > 5 AND success_rate > 0.4 ORDER BY priority DESC LIMIT 2000"
            self.associations_df = pd.read_sql_query(query, con)
        except sqlite3.Error as e:
            self.status_messages.append(f"Database error loading associations: {e}")
            self.associations_df = pd.DataFrame()
        except Exception as e:
            self.status_messages.append(f"Unexpected error loading associations: {e}")
            self.associations_df = pd.DataFrame()
        finally:
            if con:
                try:
                    con.close()
                except Exception as e:
                    self.status_messages.append(f"Warning: Could not close database connection: {e}")
        if self.associations_df.empty:
            self.status_messages.append(f"Loaded '{DB_FILE}', but found 0 strong word associations to display.")
            self.status_messages.append("(Associations need priority > 5 and success > 0.4).")
        else:
            self.status_messages.append(f"Loaded {len(self.associations_df)} strong associations from '{DB_FILE}'.")

    def _populate_graph(self):
        """Builds the NetworkX graph from loaded data."""
        if not self.clusters and self.associations_df.empty:
            return

        for cluster_id, words in self.clusters.items():
            self.graph.add_node(f"cluster_{cluster_id}", type='cluster', id=cluster_id, size=len(words))
            for word in words:
                attention = self.attention_weights.get(word, 1.0)
                self.graph.add_node(word, type='word', attention=attention)
                self.graph.add_edge(f"cluster_{cluster_id}", word, type='membership')

        for _, row in self.associations_df.iterrows():
            source, target = row['source_word'], row['next_word']
            if self.graph.has_node(source) and self.graph.has_node(target):
                self.graph.add_edge(source, target, type='association',
                                    priority=row['priority'], success=row['success_rate'])

class BrainVisualizer(QWidget):
    """A PySide6 widget to display the 3D brain graph using PyVista."""

    def __init__(self):
        super().__init__()
        self.plotter = QtInteractor(self)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.plotter.interactor)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.graph = None
        self.positions = None

    def clear_scene(self):
        try:
            self.plotter.clear()
            self.plotter.clear_actors()
            gc.collect()
        except Exception as e:
            print(f"Warning: Could not clear scene: {e}")

    def render_graph(self, graph, priority_threshold=5, show_labels=True):
        self.clear_scene()
        if not graph or graph.number_of_nodes() == 0:
            self.plotter.add_text("No data to display.\nLoad data or train the AI more.", font_size=15, position='center')
            return

        self.graph = graph

        clusters = [n for n, d in self.graph.nodes(data=True) if d['type'] == 'cluster']
        if not clusters:
            self.plotter.add_text("No clusters found to visualize.", font_size=15, position='center')
            return
            
        cluster_graph = self.graph.subgraph(clusters)
        pos_clusters = nx.spring_layout(cluster_graph, dim=3, iterations=100, scale=15.0)

        self.positions = pos_clusters.copy()
        for cluster_id, words in self.graph.adj.items():
            if self.graph.nodes[cluster_id]['type'] == 'cluster':
                cluster_center = self.positions[cluster_id]
                word_nodes = [w for w in words if self.graph.nodes[w]['type'] == 'word']
                
                num_points = len(word_nodes)
                if num_points > 0:
                    indices = np.arange(0, num_points, dtype=float) + 0.5
                    phi = np.arccos(1 - 2 * indices / num_points)
                    theta = np.pi * (1 + 5**0.5) * indices
                    radius = 2.0 + np.log1p(num_points) * 0.5
                    
                    x = cluster_center[0] + radius * np.cos(theta) * np.sin(phi)
                    y = cluster_center[1] + radius * np.sin(theta) * np.sin(phi)
                    z = cluster_center[2] + radius * np.cos(phi)

                    for i, word in enumerate(word_nodes):
                        self.positions[word] = (x[i], y[i], z[i])

        node_positions = np.array(list(self.positions.values()))
        node_types = [self.graph.nodes[n]['type'] for n in self.positions.keys()]
        
        cluster_indices = [i for i, n_type in enumerate(node_types) if n_type == 'cluster']
        if cluster_indices:
            cluster_geom = pv.PolyData(node_positions[cluster_indices])
            cluster_sizes = [self.graph.nodes[n]['size'] for n in self.positions.keys() if self.graph.nodes[n]['type'] == 'cluster']
            scaled_sizes = 5 + np.log1p(cluster_sizes) * 5
            cluster_geom['size'] = scaled_sizes
            
            self.plotter.add_mesh(
                cluster_geom,
                render_points_as_spheres=True,
                color='#007ACC',
                opacity=0.6,
                name='clusters',
                scalars='size',
                scalar_bar_args={'title': 'Cluster Size'}
            )

        word_indices = [i for i, n_type in enumerate(node_types) if n_type == 'word']
        if word_indices:
            attention_values = [d['attention'] for n, d in self.graph.nodes(data=True) if d['type'] == 'word']
            self.plotter.add_points(
                node_positions[word_indices], render_points_as_spheres=True, point_size=8,
                opacity=0.9, name='words', scalars=attention_values, cmap='viridis',
                scalar_bar_args={'title': 'Attention Weight'}
            )

        edges, edge_priorities = [], []
        for u, v, d in self.graph.edges(data=True):
            if d.get('type') == 'association' and d['priority'] >= priority_threshold:
                if u in self.positions and v in self.positions:
                    edges.append([self.positions[u], self.positions[v]])
                    edge_priorities.append(d['priority'])

        if edges:
            edge_lines = pv.MultiBlock([pv.Line(p1, p2) for p1, p2 in edges])
            for i, block in enumerate(edge_lines):
                block.cell_data["priority"] = edge_priorities[i]
            
            self.plotter.add_mesh(
                edge_lines, 
                scalars="priority", 
                cmap='coolwarm', 
                line_width=4,
                scalar_bar_args={'title': 'Association Priority'}
            )

        if show_labels and cluster_indices:
            cluster_labels = [f"Cluster {d['id']}" for n, d in self.graph.nodes(data=True) if d['type'] == 'cluster']
            self.plotter.add_point_labels(node_positions[cluster_indices], cluster_labels,
                                          font_size=12, text_color='white', shape_opacity=0.5)

        self.plotter.camera_position = 'iso'
        self.plotter.enable_zoom_style()
        self.plotter.reset_camera()

class MainWindow(QMainWindow):
    """The main application window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mai AI - 3D Brain Map Viewer")
        self.setGeometry(100, 100, 1200, 900)
        self.data_loader = None
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.control_panel = QFrame()
        self.control_panel.setFrameShape(QFrame.StyledPanel)
        self.control_panel.setFixedWidth(300)
        self.control_layout = QVBoxLayout(self.control_panel)
        file_group = QGroupBox("Data Loading")
        file_layout = QVBoxLayout(file_group)
        self.btn_load = QPushButton("Load Brain Data")
        self.btn_load.clicked.connect(self.load_data)
        self.status_label = QLabel("Status: Please load brain data.")
        self.status_label.setWordWrap(True)
        file_layout.addWidget(self.btn_load)
        file_layout.addWidget(self.status_label)
        self.control_layout.addWidget(file_group)
        vis_group = QGroupBox("Visualization Controls")
        vis_layout = QVBoxLayout(vis_group)
        self.priority_slider = QSlider(Qt.Horizontal)
        self.priority_slider.setMinimum(0)
        self.priority_slider.setMaximum(50)
        self.priority_slider.setValue(5)
        self.priority_slider.valueChanged.connect(self.update_visualization)
        self.priority_label = QLabel("Association Priority: 5")
        self.chk_labels = QCheckBox("Show Cluster Labels")
        self.chk_labels.setChecked(True)
        self.chk_labels.stateChanged.connect(self.update_visualization)
        vis_layout.addWidget(self.priority_label)
        vis_layout.addWidget(self.priority_slider)
        vis_layout.addWidget(self.chk_labels)
        self.control_layout.addWidget(vis_group)
        self.control_layout.addStretch()
        self.viewer = BrainVisualizer()
        self.main_layout.addWidget(self.control_panel)
        self.main_layout.addWidget(self.viewer, 1)

    def load_data(self):
        """Open a dialog to select the directory and load the data."""
        path = QFileDialog.getExistingDirectory(self, "Select Directory Containing Brain Files")
        if path:
            self.status_label.setText("Status: Loading data...")
            QApplication.processEvents() 

            self.data_loader = BrainDataLoader(path)
            self.data_loader.load_data()
            
            status_text = "\n".join(self.data_loader.status_messages)
            self.status_label.setText(status_text)

            if "Error" in status_text:
                QMessageBox.critical(self, "Loading Error", status_text)
            
            self.update_visualization()

    def update_visualization(self):
        """Rerenders the 3D scene based on current control settings."""
        if self.data_loader and self.data_loader.graph is not None:
            priority = self.priority_slider.value()
            self.priority_label.setText(f"Association Priority: {priority}")
            show_labels = self.chk_labels.isChecked()
            self.viewer.render_graph(self.data_loader.graph, priority_threshold=priority, show_labels=show_labels)
        else:
            self.viewer.render_graph(None)
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
