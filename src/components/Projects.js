import React from 'react'
import { Tilt } from 'react-tilt';
import { motion } from 'framer-motion';
import { styles } from '../constants/styles';
import { github, link } from '../assets';
import SectionWrapper from './SectionWrapper';
import { projects } from '../constants/info';
import { fadeIn, textVariant, textVariantPhone } from '../constants/motion';
import { useMediaQuery } from 'react-responsive';

const ProjectCard = ({name, index, description, tags, image, source_code_link, isGithub}) => {
  const isMobile = useMediaQuery({maxWidth: 955});
  return isMobile ? (
        <div
          data-aos = "zoom-in-up"
          data-aos-duration = {400}
          data-aos-delay = {200}
        >
          <Tilt
            options = {{
              max: 45,
              scale: 1,
              speed: 450
            }}
            className = "bg-tertiary p-5 rounded-2xl sm:w-[360px] w-full"
          >
            <div className="relative w-full h-[230px] ">
              <img 
                src={image} 
                alt={name} 
                className='w-full h-full object-cover rounded-2xl'
              />
              <div className="absolute inset-0 flex justify-end m-3 card-img_hover">
                <div 
                  className="black-gradient w-10 h-10 rounded-full flex items-center justify-center cursor-pointer" 
                  onClick={()=> window.open(source_code_link, "_blank")}
                >
                  <img 
                    src={isGithub ? github : link} 
                    alt="githib" 
                    className='w-1/2 h-1/2 object-contain'
                  />
                </div>
              </div>
              
            </div>

            <div className="mt-5">
              <h3 className="text-white font-bold text-[24px]">{name}</h3>
              <p className='mt-2 text-secondary text-[14px]'>{description}</p>
            </div>
            
            <div className="mt-4 flex flex-wrap gap-2">
              {tags.map((tag)=>(
                <p key={tag.name} className={`text-[14px] ${tag.color}`}>#{tag.name}</p>
              ))}
            </div>
          </Tilt>
        </div>
      ) : (
        <motion.div
          variants={fadeIn("up", "spring", index * 0.15, 0.75)}
        >
          <Tilt
            options = {{
              max: 45,
              scale: 1,
              speed: 450
            }}
            className = "bg-tertiary p-5 rounded-2xl sm:w-[360px] w-full"
          >
            <div className="relative w-full h-[230px] ">
              <img 
                src={image} 
                alt={name} 
                className='w-full h-full object-cover rounded-2xl'
              />
              <div className="absolute inset-0 flex justify-end m-3 card-img_hover">
                <div 
                  className="black-gradient w-10 h-10 rounded-full flex items-center justify-center cursor-pointer" 
                  onClick={()=> window.open(source_code_link, "_blank")}
                >
                  <img 
                    src={isGithub ? github : link} 
                    alt="githib" 
                    className='w-1/2 h-1/2 object-contain'
                  />
                </div>
              </div>
              
            </div>

            <div className="mt-5">
              <h3 className="text-white font-bold text-[24px]">{name}</h3>
              <p className='mt-2 text-secondary text-[14px]'>{description}</p>
            </div>
            
            <div className="mt-4 flex flex-wrap gap-2">
              {tags.map((tag)=>(
                <p key={tag.name} className={`text-[14px] ${tag.color}`}>#{tag.name}</p>
              ))}
            </div>
          </Tilt>
        </motion.div>
  )
}

const Works = () => {
  const isMobile = useMediaQuery({maxWidth: 955});

  return (
    <>
      {isMobile ? (
         <div
          variants={isMobile ? textVariantPhone() : null}
          data-aos = "fade-down"
          data-aos-duration = {500}
        >
          <p className={styles.sectionSubText}>My work</p>
          <h2 className={styles.sectionHeadText}>Projects.</h2>
        </div>
      ) : (
        <motion.div
          variants={isMobile ? textVariantPhone() : null}
        >
          <p className={styles.sectionSubText}>My work</p>
          <h2 className={styles.sectionHeadText}>Projects.</h2>
        </motion.div>
      )}
      <div className="mt-10 flex flex-wrap gap-7">
        {projects.map((project, index)=>(
          <ProjectCard 
            key = {`project-${index}`} 
            index = {index} 
            {...project} 
          />
        ))}
      </div>
    </>
  )
}

export default SectionWrapper(Works, "projects")