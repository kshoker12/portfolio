import logo from './logo.svg';
import './App.css';
import React, { useEffect } from 'react'
import { Provider } from './context/context';
import * as HELPER from './helpers/appHelper';
import Landing from './components/Landing';

/**
 * @description App parent component which houses all portfolio components
 * @returns {Element}
 */
const App = () => {
  /** Initialize AOS library which can be reused throughout the portfolio for animated components */
  useEffect(()=>{
    HELPER.initializeAOS();
  },[]);

  /** All app components such as navigation bar */
  return (
    <Provider>
      <Landing/>
    </Provider>
  );
};

export default App;
