import logo from './logo.svg';
import './App.css';
import React, { useEffect, useRef, useState } from 'react'
import { Provider } from './context/context';
import * as HELPER from './helpers/appHelper';
import Landing from './components/Landing';
import About from './components/About';
import WorkExperience from './components/WorkExperience';
import Projects from './components/Projects';
import Testimonials from './components/Testimonials';
import Contact from './components/Contact';
import Skills from './components/Skills';
import { navLinks } from './constants';

/**
 * @description App parent component which houses all portfolio components
 * @returns {Element}
 */
const App = () => {
  /** Indicator for turning on navigation bar background */
  const [showColor, setShowColor] = useState(false);
  /** Indicator of active tab */
  const [active, setActive] = useState('');
  /** Reference for about us section */
  const aboutUsRef = useRef();

  /** Initialize AOS library which can be reused throughout the portfolio for animated components */
  useEffect(()=>{
    HELPER.initializeAOS();
  },[]);

  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.scrollY + window.innerHeight / 2; // Middle of the viewport

      for (const link of navLinks) {
        const section = document.getElementById(link.id);
        if (section) {
          const sectionTop = section.offsetTop;
          const sectionBottom = sectionTop + section.offsetHeight;

          if (scrollPosition >= sectionTop && scrollPosition <= sectionBottom) {
            setActive(link.id); // Update active section
            break;
          }
        }
      }
    };

    window.addEventListener("scroll", handleScroll);

    // Cleanup listener on component unmount
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  /** Add behavior watching for scroll into about us */
  useEffect(() => {
    const handleScroll = () => {
      const target = aboutUsRef.current;
      if (target) {
            const targetPosition = target.getBoundingClientRect().top + window.innerHeight / 1.2;
            const viewportHeight = window.innerHeight -180;
            
            const newBool = targetPosition <= viewportHeight
            setShowColor(newBool);
        }
    };

    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  /** All app components such as navigation bar */
  return (
    <Provider>
      <div className='bg-darker relative' ref={aboutUsRef}>
        <Landing showColor={showColor} active={active} setActive={setActive}/>
        <About />
        <Skills/>
        <WorkExperience/>
        <Projects/>
        <Testimonials/>
        <Contact/>
        <div className='fixed bottom-0 right-0 rounded-full bg-purple-500 px-2 py-1 m-2 hover:px-3 hover:py-2 hover:bg-primarydark ease-in-out duration-200'>
          <a href='#'>
            <i className='fa fa-arrow-up text-3xl text-white'/>
          </a>
        </div>
      </div>
    </Provider>
  );
};

export default App;
