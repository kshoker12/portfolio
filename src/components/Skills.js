import React from 'react'
import { Tilt } from 'react-tilt'
import { motion } from 'framer-motion'
import { styles } from '../constants/styles'
import { certifications, languages, mlAlgos } from '../constants/info'
import { fadeIn, textVariant, textVariantPhone } from "../constants/motion"
import SectionWrapper from './SectionWrapper'

const ServiceCard = ({index, name, icon}) => {
  return (
    <Tilt className = "w-[130px]">
      <motion.div 
        variants={fadeIn("right", "spring", 0.15 * index, 0.75)}
        className='w-[130px] green-pink-gradient p-[1px] rounded-[20px] shadow-card'
      >
        <div 
          className="bg-tertiary rounded-[20px] py-5 px-12 min-h-[130px] flex justify-evenly items-center flex-col"
          options = {{
            max: 45,
            scale: 1,
            speed: 1000
          }}
        >
          <img src={icon} alt='title'className='w-12 h-12 object-contain'/>
          <h3 className="text-white font-semibold text-center text-sm">
            {name}
          </h3>

        </div>
      </motion.div>
    </Tilt>
  )
}
 
const Skills = () => {
  return (
    <>
      <motion.div
          variants={textVariantPhone()}
        >
          <p className={styles.sectionSubText}>Technical Skills</p>
          <h2 className={styles.sectionHeadText}>Software Tools.</h2>
        </motion.div>
        <div className="mt-10 flex flex-wrap gap-10">
          {languages.map((skill, index)=>{
            return (
              <ServiceCard key = {skill.name} index = {index} {...skill} />
            )
          })}
        </div>
      </>   
  )
}

export default SectionWrapper(Skills, "skills")