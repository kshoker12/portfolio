import React, { useState } from 'react'
import { styles } from '../constants/styles';
import { navLinks } from '../constants';
import { logo, menu, close } from '../images';
import Scrollspy from 'react-scrollspy';

/**
 * @description Navigation bar which includes a linkage to all sections of the application
 * @returns {Element}
 */
const Navigation = ({showColor, active, setActive}) => {
  const [toggle, setToggle] = useState(false);
  const [updateCount, setUpdateCount] = useState(1);

  console.log(active);

  /** Render navigation bar of sections in application */
  return (
    <Scrollspy
        items={["about", "skills", "experience", "projects", "testimonials", "contact"]}
        currentClassName="active"
        onUpdate={(el)=>{
            if (el != undefined && el?.id ) setActive(el.id);
            setUpdateCount(updateCount+1);
        }}
    >
        <nav
        className={`w-full flex items-center py-5 fixed top-0 z-20`}
        >
            <div className={`w-screen h-full absolute -z-10 ${showColor && 'bg-tertiary opacity-80'}`}/>
            <div className={`${styles.paddingX} flex w-full justify-between items-center max-w-8xl mx-auto`}>
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
                        <a href={`#${nav.id}`}>
                            <li 
                                key={nav.id}
                                className = {`transition-underline inline-flex items-center px-1 pr-1 pb-2 font-bold text-md leading-5 transition duration-150 ease-in-out focus:outline-none ${
                                    active === nav.id ? 'active  text-white  hover:text-purple-200' : 'border-transparent text-gray-100 hover:text-purple-100 focus:text-purple-100'
                                }`}
                                onClick={()=>setActive(nav.id)}
                            >
                                {nav.title}
                            </li>
                        </a>
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
                    
                    <div className={`${!toggle ? "hidden" : "flex"} p-6 black-gradient absolute top-20 right-0 mx-4 my-2 min-w-[140px] z-10 rounded-xl border-[1px] border-white`}>
                        <ul className="list-none flex justify-end items-start flex-col gap-4">
                            {navLinks.map((nav)=>{
                                return (
                                <li 
                                    key={nav.id}
                                    className={`${nav.title === active ? "text-purple-200 font-semibold" : "text-secondary"} font-poppins font-medium cursor-pointer text-[16px] hover:text-purple-200 hover:font-semibold ease-in-out duration-200`}
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
    </Scrollspy>
    
  )
}

export default Navigation