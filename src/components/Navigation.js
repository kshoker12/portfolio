import React, { useState } from 'react'
import { styles } from '../constants/styles';
import { navLinks } from '../constants';
import { logo, menu, close } from '../images';

/**
 * @description Navigation bar which includes a linkage to all sections of the application
 * @returns {Element}
 */
const Navigation = () => {
  const [active, setActive] = useState('');
  const [toggle, setToggle] = useState(false);

  /** Render navigation bar of sections in application */
  return (
    <nav
      className={`${styles.paddingX} w-full flex items-center py-5 fixed top-0 z-20 `}
    >
      <div className="flex w-full justify-between items-center max-w-8xl mx-auto">
        <div className=''>
            <a 
            to="/"
            className="flex items-center gap-2"
            onClick={()=>{
                setActive("");
                window.scrollTo(0, 0);
            }}
            >
                <p className="text-white text-[18px] font-bold cursor-pointer flex">
                    Karandeep&nbsp;<span className='block'>| Shoker</span>
                </p>
            </a>
            <div className='flex items-center justify-between'>
                <a className='' href='https://www.linkedin.com/in/kshoker12/' target='_blank'>
                    <i className='bx bxl-linkedin text-white text-3xl hover:text-purple-400 ease-in-out duration-200'/>  
                </a>
                <a className='' href='https://github.com/kshoker12' target='_blank'>
                    <i className='bx bxl-github text-white text-3xl hover:text-purple-400 ease-in-out duration-200'/>      
                </a>
                <a className='' href='mailto:karandeep.shoker@outlook.com' target='_blank'>
                    <i className='bx bxs-envelope text-white text-3xl hover:text-purple-400 ease-in-out duration-200'/>      
                </a>
                <a href="tel:+17786814099" className=''>
                    <i className='bx bxs-phone text-white text-3xl hover:text-purple-400 ease-in-out duration-200'/>   
                </a>    
            </div>    
        </div>
        <ul className="list-none hidden lg:flex flex-row gap-10">
            {navLinks.map((nav)=>{
                return (
                <li 
                    key={nav.id}
                    className = {`inline-flex items-center px-1 pr-1 pb-2 font-bold text-md leading-5 transition duration-150 ease-in-out focus:outline-none ${
                        active === nav.title ? 'border-b-[3px] border-purple-400 text-white focus:border-purple-400 hover:text-purple-200' : 'transition-underline border-transparent text-gray-200 hover:text-purple-100 focus:text-purple-100'
                    }`}
                    onClick={()=>setActive(nav.title)}
                >
                    <a href={`#${nav.id}`}>{nav.title}</a>
                </li>
                )
            })}
        </ul>

        <div className="lg:hidden flex flex-1 justify-end items-center">
            <img 
                src={toggle ? close : menu}
                alt="menu" 
                className='w-[28px] h-[28px] object-contain cursor-pointer'
                onClick={()=>setToggle(!toggle)}
            />
            
            <div className={`${!toggle ? "hidden" : "flex"} p-6 black-gradient absolute top-20 right-0 mx-4 my-2 min-w-[140px] z-10 rounded-xl`}>
                <ul className="list-none flex justify-end items-start flex-col gap-4">
                    {navLinks.map((nav)=>{
                        return (
                        <li 
                            key={nav.id}
                            className={`${nav.title === active ? "text-white" : "text-secondary"} font-poppins font-medium cursor-pointer text-[16px]`}
                            onClick={()=>{
                            setToggle(!toggle);
                            setActive(nav.title); 
                            }}
                        >
                            <a href={`#${nav.id}`}>{nav.title}</a>
                        </li>
                        )
                    })}
                </ul>
            </div>
        </div>
      </div>
    </nav>
  )
}

export default Navigation