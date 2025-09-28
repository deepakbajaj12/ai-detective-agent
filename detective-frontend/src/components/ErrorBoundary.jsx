import React from 'react';

export default class ErrorBoundary extends React.Component {
  constructor(props){
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error){
    return { hasError: true, error };
  }
  componentDidCatch(error, info){
    // eslint-disable-next-line no-console
    console.error('UI ErrorBoundary caught', error, info);
  }
  render(){
    if(this.state.hasError){
      return (
        <div style={{ padding: '2rem', fontFamily: 'Merriweather, serif' }}>
          <h2 style={{ marginTop:0 }}>Interface Error</h2>
          <p>Something went wrong while rendering the interface.</p>
          {this.state.error && (
            <pre style={{ background:'#2e2e2e', color:'#eee', padding:'1rem', borderRadius:8, overflowX:'auto', maxHeight:220 }}>{String(this.state.error)}</pre>
          )}
          <button onClick={()=> window.location.reload()}>Reload</button>
        </div>
      );
    }
    return this.props.children;
  }
}
