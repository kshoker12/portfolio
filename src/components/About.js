import React from 'react'
import { Tilt } from 'react-tilt'
import { motion } from 'framer-motion'
import { styles } from '../constants/styles'
import { aboutMe } from '../constants/info'
import { fadeIn, textVariant } from "../constants/motion"
import SectionWrapper from './SectionWrapper'
import selfPic from '../images/self2.jpeg'

const About = () => {
  return (
    <>
      <motion.div variants={textVariant()}>
        <p className={styles.sectionSubText}>Introduction</p>
        <h2 className={styles.sectionHeadText}>About me.</h2>
      </motion.div>
      <div className='lg:flex lg:space-x-6'>
        <motion.p 
          variants={fadeIn("", "", 0.1, 1)}
          className='mt-4 text-secondarydark text-[17px] max-w-3xl leading-[30px]'
        >
          {aboutMe.map((paragraph)=>(
            <div className='mb-1'>
              {paragraph}
            </div>
          ))}
        </motion.p>
        <img className='xl:w-[340px] xl:h-[384px] lg:w-[300px] lg:h-[338px] w-[250px] h-[280px] shadow-xl shadow-black mt-5' src={selfPic} data-aos = "fade-left" data-aos-delay = {100}/>  
      </div>
      
    </>
  )
}

export default SectionWrapper(About, "about")