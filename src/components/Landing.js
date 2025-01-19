import React, { Suspense } from 'react'
import mainbg from '../images/mainbg.jpg'
import Navigation from './Navigation';
import { styles } from '../constants/styles';
import { motion } from 'framer-motion';
import self from '../images/self.png';
import mainbg1 from '../images//mainbg2.webp'
import About from './About';

/**
 * @description Main Landing page of the application highlighting important information
 * @returns {Element}
 */
const Landing = ({showColor, active, setActive}) => {

    /** Render background with 3d Model */
    return (
        <div 
            className='h-screen w-screen bg-center bg-cover'
            style={{ backgroundImage: `url(${mainbg1})` }}
        >
            <div className='w-full h-full absolute opacity-10 bg-purple-500'/>
            <div className='w-full h-full absolute opacity-30 bg-black'/>
            <Navigation showColor={showColor} active={active} setActive={setActive}/>
            <section className="relative w-full h-screen mx-auto">
                <div className={`${styles.paddingX} absolute w-full top-[120px] mx-auto flex md:flex-row flex-col md:justify-center space-x-6`} >
                    <div className='flex flex-row md:items-start justify-start md:justify-center gap-5 items-center'>
                        <div className="flex flex-col justify-center items-center mt-5" data-aos = "fade-right" data-aos-delay = {100} >
                            <div className="w-5 h-5 rounded-full bg-purple-400"/>
                            <div className="w-1 sm:h-44 h-20 md:h-40 bg-purple-400"/>
                        </div>
                        <div className="" data-aos = "fade-right" data-aos-delay = {100}>
                            <h1 className={`${styles.heroHeadText} text-white`}>
                                Hi, I'm <span className='text-blue-300'>Karandeep </span>
                            </h1>
                            <p className={`${styles.heroSubText} text-white mt-2 xl:max-w-3xl lg:max-w-2xl max-w-xl`}>
                                I like developing innovative software & Machine <br className='sm:block hidden'/>Learning solutions to tackle complex real-world problems
                            </p>
                        </div>    
                    </div>
                    <img className='xl:w-[340px] xl:h-[384px] lg:w-[300px] lg:h-[338px] md:w-[250px] md:h-[280px] w-[200px] h-[220px] shadow-xl shadow-black mt-5' src={self} data-aos = "fade-left" data-aos-delay = {100}/>
                </div>
                <div className="absolute xs:bottom-10 bottom-32 w-full flex justify-center items-center">
                    <a href="#about">
                    <div className="w-[35px] h-[64px] rounded-3xl border-4 border-secondary flex justify-center items-start p-2">
                        <motion.dev
                            animate = {{
                                y: [0, 24, 0]
                            }}
                            transition = {{
                                duration: 1.5,
                                repeat: Infinity,
                                repeatType: "loop"
                            }}
                            className = "w-6 h-4 rounded-full bg-purple-200 mb-1"
                        />
                    </div>
                    </a>
                </div>
            </section>
        </div>
    );
};

export default Landing;