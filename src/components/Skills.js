import React from 'react'
import { Tilt } from 'react-tilt'
import { motion } from 'framer-motion'
import { styles } from '../constants/styles'
import { certifications, languages, mlAlgos } from '../constants/info'
import { fadeIn, textVariant } from "../constants/motion"
import SectionWrapper from './SectionWrapper'

const ServiceCard = ({index, name, icon}) => {
  return (
    // <Tilt className = "w-[130px]">
      <motion.div 
        // variants={fadeIn("right", "spring", 0.15 * index, 0.75)}
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
    // </Tilt>
  )
}

const ServiceCardAlt = ({index, name, icon, certificate, link}) => {
  return (
    // <Tilt className = "w-[250px]">
      <motion.div 
        // variants={fadeIn("right", "spring", 0.15 * index, 0.75)}
        className='w-[250px] green-pink-gradient p-[1px] rounded-[20px] shadow-card'
      >
        <div 
          className="bg-tertiary rounded-[20px] py-5 px-6 min-h-[130px] flex justify-evenly items-center flex-col"
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
          <a href={link} target='_blank'>
            <img src = {certificate} alt='title' className='object-contain w-40 h-32'/>
          </a>
        </div>
      </motion.div>
    // </Tilt>
  )
}
 
const Skills = () => {
  return (
    <div className='space-y-6'>
      <motion.div variants={textVariant()}>
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
      <motion.div variants={textVariant()}>
        <h2 className={styles.sectionHeadText}>Machine-Learning Algorithms.</h2>
      </motion.div>
      <div className="mt-10 flex flex-wrap gap-10">
        {mlAlgos.map((skill, index)=>{
          return (
            <ServiceCard key = {skill.name} index = {index} {...skill} />
          )
        })}
      </div>
      <motion.div variants={textVariant()}>
        <h2 className={styles.sectionHeadText}>Certifications.</h2>
      </motion.div>
      <div className="mt-10 flex flex-wrap gap-10">
        {certifications.map((skill, index)=>{
          return (
            <ServiceCardAlt key = {skill.name} index = {index} {...skill} />
          )
        })}
      </div>
    </div>
  )
}

export default SectionWrapper(Skills, "skills")