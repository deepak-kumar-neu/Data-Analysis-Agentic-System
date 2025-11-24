"""
Real-time processing components for Streamlit UI.
Handles streaming updates and live feedback during analysis.
"""

import streamlit as st
import time
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime
import queue
import threading


class ProcessingStream:
    """Manages real-time processing updates."""
    
    def __init__(self):
        """Initialize processing stream."""
        self.message_queue = queue.Queue()
        self.is_running = False
        self.current_step = ""
        self.progress = 0.0
        
    def start(self):
        """Start the processing stream."""
        self.is_running = True
        self.message_queue = queue.Queue()
        
    def stop(self):
        """Stop the processing stream."""
        self.is_running = False
        
    def update(self, message: str, progress: float, agent: str = "System", status: str = "info"):
        """
        Add update to stream.
        
        Args:
            message: Update message
            progress: Progress percentage (0-100)
            agent: Agent name
            status: Status type (info, success, warning, error)
        """
        if self.is_running:
            update_data = {
                'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3],
                'message': message,
                'progress': progress,
                'agent': agent,
                'status': status
            }
            self.message_queue.put(update_data)
            self.current_step = message
            self.progress = progress
            
    def get_updates(self) -> List[Dict[str, Any]]:
        """Get all pending updates."""
        updates = []
        while not self.message_queue.empty():
            try:
                updates.append(self.message_queue.get_nowait())
            except queue.Empty:
                break
        return updates


class LiveProgressBar:
    """Live updating progress bar component."""
    
    def __init__(self, container):
        """
        Initialize progress bar.
        
        Args:
            container: Streamlit container to render in
        """
        self.container = container
        self.progress_bar = None
        self.status_text = None
        
    def render(self, progress: float, message: str = ""):
        """
        Render/update progress bar.
        
        Args:
            progress: Progress value (0-100)
            message: Status message
        """
        with self.container:
            if self.progress_bar is None:
                self.progress_bar = st.progress(0)
                self.status_text = st.empty()
                
            self.progress_bar.progress(min(progress / 100, 1.0))
            if message:
                self.status_text.text(f"⚡ {message}")
                
    def complete(self, message: str = "Complete!"):
        """Mark progress as complete."""
        self.render(100, message)
        
    def error(self, message: str = "Error occurred"):
        """Mark progress with error."""
        with self.container:
            if self.status_text:
                self.status_text.error(f"❌ {message}")


class AgentStatusMonitor:
    """Monitor and display agent statuses in real-time."""
    
    def __init__(self):
        """Initialize agent monitor."""
        self.agent_states = {}
        
    def update_agent(self, agent_id: str, state: str, task: str = "", progress: float = 0):
        """
        Update agent status.
        
        Args:
            agent_id: Agent identifier
            state: Current state (idle, running, completed, error)
            task: Current task description
            progress: Task progress (0-100)
        """
        self.agent_states[agent_id] = {
            'state': state,
            'task': task,
            'progress': progress,
            'last_update': datetime.now()
        }
        
    def render(self, container):
        """
        Render agent status monitor.
        
        Args:
            container: Streamlit container
        """
        with container:
            st.markdown("### 🤖 Agent Monitor")
            
            agents = {
                'controller': ('🎯', 'Controller'),
                'data_collection': ('📥', 'Data Collection'),
                'data_processing': ('⚙️', 'Processing'),
                'analysis': ('🔬', 'Analysis'),
                'visualization': ('📊', 'Visualization'),
                'quality_assurance': ('✓', 'QA')
            }
            
            cols = st.columns(3)
            
            for idx, (agent_id, (icon, name)) in enumerate(agents.items()):
                with cols[idx % 3]:
                    status = self.agent_states.get(agent_id, {})
                    state = status.get('state', 'idle')
                    task = status.get('task', 'Waiting...')
                    progress = status.get('progress', 0)
                    
                    # State emoji
                    state_emoji = {
                        'idle': '⚪',
                        'running': '🟡',
                        'completed': '🟢',
                        'error': '🔴'
                    }.get(state, '⚪')
                    
                    st.markdown(
                        f"""
                        <div style='padding: 1rem; background: #f8f9fa; 
                                    border-radius: 0.5rem; margin: 0.5rem 0;
                                    border-left: 4px solid #667eea;'>
                            <h4>{icon} {name}</h4>
                            <p style='margin: 0.5rem 0;'>{state_emoji} {state.title()}</p>
                            <small>{task}</small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    if state == 'running' and progress > 0:
                        st.progress(progress / 100)


class ExecutionTimeline:
    """Display execution timeline with events."""
    
    def __init__(self):
        """Initialize timeline."""
        self.events = []
        
    def add_event(self, agent: str, action: str, status: str = "info", details: str = ""):
        """
        Add event to timeline.
        
        Args:
            agent: Agent name
            action: Action description
            status: Event status (info, success, warning, error)
            details: Additional details
        """
        event = {
            'timestamp': datetime.now(),
            'agent': agent,
            'action': action,
            'status': status,
            'details': details
        }
        self.events.append(event)
        
    def render(self, container, max_events: int = 20):
        """
        Render timeline.
        
        Args:
            container: Streamlit container
            max_events: Maximum events to display
        """
        with container:
            st.markdown("### 📝 Execution Timeline")
            
            # Show recent events
            recent_events = self.events[-max_events:]
            
            for event in reversed(recent_events):
                timestamp = event['timestamp'].strftime("%H:%M:%S")
                agent = event['agent']
                action = event['action']
                status = event['status']
                
                # Icon based on status
                icon = {
                    'info': 'ℹ️',
                    'success': '✅',
                    'warning': '⚠️',
                    'error': '❌'
                }.get(status, 'ℹ️')
                
                # Color based on status
                color = {
                    'info': '#17a2b8',
                    'success': '#28a745',
                    'warning': '#ffc107',
                    'error': '#dc3545'
                }.get(status, '#17a2b8')
                
                st.markdown(
                    f"""
                    <div style='padding: 0.5rem; margin: 0.3rem 0;
                                border-left: 3px solid {color};
                                padding-left: 1rem; background: #f8f9fa;'>
                        {icon} <strong>{timestamp}</strong> | {agent}: {action}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


class MetricsDisplay:
    """Display real-time metrics and KPIs."""
    
    def __init__(self):
        """Initialize metrics display."""
        self.metrics = {}
        
    def update_metric(self, name: str, value: Any, delta: Optional[Any] = None, 
                     help_text: str = ""):
        """
        Update a metric.
        
        Args:
            name: Metric name
            value: Current value
            delta: Change from previous value
            help_text: Help text
        """
        self.metrics[name] = {
            'value': value,
            'delta': delta,
            'help': help_text,
            'updated': datetime.now()
        }
        
    def render(self, container):
        """
        Render metrics.
        
        Args:
            container: Streamlit container
        """
        with container:
            if not self.metrics:
                st.info("No metrics available yet")
                return
                
            # Display in columns
            metric_items = list(self.metrics.items())
            cols = st.columns(min(len(metric_items), 4))
            
            for idx, (name, data) in enumerate(metric_items):
                with cols[idx % 4]:
                    st.metric(
                        label=name,
                        value=data['value'],
                        delta=data.get('delta'),
                        help=data.get('help', '')
                    )


class VisualizationStream:
    """Stream visualizations as they're generated."""
    
    def __init__(self):
        """Initialize visualization stream."""
        self.visualizations = []
        
    def add_visualization(self, viz_type: str, data: Any, config: Dict[str, Any], 
                         title: str = ""):
        """
        Add visualization to stream.
        
        Args:
            viz_type: Type of visualization
            data: Visualization data
            config: Configuration
            title: Title
        """
        viz = {
            'type': viz_type,
            'data': data,
            'config': config,
            'title': title,
            'created': datetime.now()
        }
        self.visualizations.append(viz)
        
    def render(self, container):
        """
        Render all visualizations.
        
        Args:
            container: Streamlit container
        """
        with container:
            if not self.visualizations:
                st.info("No visualizations generated yet")
                return
                
            for viz in self.visualizations:
                st.markdown(f"#### {viz['title']}")
                
                # Render based on type
                viz_type = viz['type']
                
                if viz_type == 'dataframe':
                    st.dataframe(viz['data'], **viz.get('config', {}))
                elif viz_type == 'chart':
                    st.plotly_chart(viz['data'], **viz.get('config', {}))
                elif viz_type == 'metric':
                    st.metric(**viz['data'])
                elif viz_type == 'json':
                    st.json(viz['data'])
                elif viz_type == 'markdown':
                    st.markdown(viz['data'])


def create_live_processor(orchestrator, config: Dict[str, Any]):
    """
    Create a live processing wrapper around orchestrator.
    
    Args:
        orchestrator: Orchestrator instance
        config: Configuration
        
    Returns:
        Function that executes with live updates
    """
    stream = ProcessingStream()
    monitor = AgentStatusMonitor()
    timeline = ExecutionTimeline()
    
    def process_with_updates(data_source: str, objective: str):
        """Process with live updates."""
        stream.start()
        
        try:
            # Simulate step-by-step execution
            steps = [
                ('controller', 'Planning workflow', 10),
                ('data_collection', 'Collecting data', 25),
                ('data_processing', 'Cleaning data', 40),
                ('analysis', 'Analyzing data', 60),
                ('visualization', 'Creating visualizations', 80),
                ('quality_assurance', 'Validating results', 95)
            ]
            
            for agent_id, action, progress in steps:
                stream.update(action, progress, agent_id, 'info')
                monitor.update_agent(agent_id, 'running', action, progress)
                timeline.add_event(agent_id, action, 'info')
                time.sleep(0.5)  # Simulate work
                
            # Execute actual workflow
            results = orchestrator.execute_workflow(
                data_source=data_source,
                objective=objective,
                mode=config.get('execution_mode', 'sequential')
            )
            
            stream.update("Analysis complete", 100, 'System', 'success')
            timeline.add_event('System', 'Analysis completed', 'success')
            
            return results
            
        except Exception as e:
            stream.update(f"Error: {str(e)}", 0, 'System', 'error')
            timeline.add_event('System', f'Error occurred: {str(e)}', 'error')
            raise
            
        finally:
            stream.stop()
            
    return process_with_updates, stream, monitor, timeline
