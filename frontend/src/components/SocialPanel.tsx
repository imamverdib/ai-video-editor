import { useState } from 'react'
import { api } from '../api'
import type { Project } from '../types'

interface Props {
  project: Project
  onProjectUpdate: (p: Project) => void
  onError: (err: unknown) => void
}

export default function SocialPanel({ project, onProjectUpdate, onError }: Props) {
  const [activeTab, setActiveTab] = useState<'instagram' | 'youtube' | 'linkedin'>('instagram')
  const [generatingText, setGeneratingText] = useState(false)
  const [generatingCover, setGeneratingCover] = useState(false)
  const [publishing, setPublishing] = useState(false)
  const [scheduleTime, setScheduleTime] = useState<string>('')

  const [coverTitle, setCoverTitle] = useState<string>('')

  const handleGenerateText = async () => {
    setGeneratingText(true)
    try {
      const result = await api.generateSocialText(project.id)
      onProjectUpdate({ ...project, social_posts: result.social_posts })
    } catch (error) {
      onError(error)
    } finally {
      setGeneratingText(false)
    }
  }

  const handleGenerateCover = async () => {
    setGeneratingCover(true)
    try {
      const result = await api.generateSocialCover(project.id, coverTitle || undefined)
      onProjectUpdate({ ...project, social_cover_image: result.cover_image })
    } catch (error) {
      onError(error)
    } finally {
      setGeneratingCover(false)
    }
  }

  const handlePublish = async (platform: string) => {
    setPublishing(true)
    try {
      const timeNum = scheduleTime ? new Date(scheduleTime).getTime() / 1000 : undefined
      await api.publishSocialPost(project.id, platform, timeNum)
      
      // Update local state mock
      const newPosts = { ...project.social_posts }
      if (newPosts[platform]) {
        newPosts[platform] = { ...newPosts[platform], status: timeNum ? 'scheduled' : 'published' }
      }
      onProjectUpdate({ ...project, social_posts: newPosts })
    } catch (error) {
      onError(error)
    } finally {
      setPublishing(false)
    }
  }

  const currentPost = project.social_posts?.[activeTab]

  return (
    <div className="panel" style={{ marginTop: '1rem', flex: 1 }}>
      <div className="row" style={{ justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <h2>Share & Schedule</h2>
        <div className="row" style={{ gap: '0.5rem' }}>
          <button 
            className="ghost" 
            onClick={() => void handleGenerateText()}
            disabled={generatingText}
          >
            {generatingText ? 'Yazılır...' : '✨ Mətnləri Yaz'}
          </button>
          <button 
            className="ghost" 
            onClick={() => void handleGenerateCover()}
            disabled={generatingCover}
          >
            {generatingCover ? 'Hazırlanır...' : '🖼️ Cover Yarat'}
          </button>
        </div>
      </div>

      <div className="share-layout">
        <div className="share-cover-column">
          <div className="field" style={{ marginBottom: '1rem' }}>
            <label>Cover Üzərindəki Yazı (Sarı etmək üçün *söz* yazın)</label>
            <input 
              type="text"
              value={coverTitle}
              onChange={(e) => setCoverTitle(e.target.value)}
              placeholder="12 saatda *AI* ilə MƏHSUL yaratdım..."
              style={{ padding: '0.5rem', fontSize: '1rem', width: '100%', boxSizing: 'border-box' }}
            />
          </div>
          {project.social_cover_image ? (
            <div className="field">
              <label>Cover Şəkli</label>
              <img 
                src={`/api/projects/${project.id}/output?name=${project.social_cover_image}`} 
                alt="Cover" 
                style={{ width: '100%', borderRadius: '4px', border: '1px solid var(--border)', marginTop: '0.5rem', objectFit: 'contain' }}
              />
            </div>
          ) : (
            <div className="field" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '400px', border: '1px dashed var(--border)', borderRadius: '4px' }}>
              <span className="muted" style={{opacity: 0.5}}>Cover hələ yaradılmayıb</span>
            </div>
          )}
        </div>

        <div className="share-text-column">
          <div className="tabs" style={{ marginBottom: '1rem', borderBottom: '1px solid var(--border)', display: 'flex', gap: '1.5rem' }}>
            {(['instagram', 'youtube', 'linkedin'] as const).map((tab) => (
              <button
                key={tab}
                className={`tab ${activeTab === tab ? 'active' : ''}`}
                onClick={() => setActiveTab(tab)}
                style={{ 
                  background: 'none', 
                  border: 'none', 
                  borderBottom: activeTab === tab ? '2px solid var(--accent)' : '2px solid transparent',
                  padding: '0.5rem 0',
                  cursor: 'pointer',
                  color: activeTab === tab ? 'var(--fg)' : 'var(--muted)',
                  textTransform: 'capitalize',
                  fontSize: '1.1rem',
                  fontWeight: 500
                }}
              >
                {tab}
              </button>
            ))}
          </div>

          <div className="field">
            <label>Açıqlama Mətni ({activeTab})</label>
            <textarea 
              rows={12}
              value={currentPost?.text || ''}
              placeholder="Mətn hələ yazılmayıb. Yuxarıdan 'Mətnləri Yaz' düyməsinə basın."
              readOnly
              style={{ fontSize: '0.95rem', width: '100%', boxSizing: 'border-box' }}
            />
          </div>

          <div className="field" style={{ marginTop: '1rem' }}>
            <label>Planlama Vaxtı (Opsional)</label>
            <input 
              type="datetime-local" 
              value={scheduleTime}
              onChange={(e) => setScheduleTime(e.target.value)}
              style={{ padding: '0.5rem', fontSize: '1rem', width: '100%', boxSizing: 'border-box' }}
            />
          </div>

          <button 
            className="primary"
            onClick={() => void handlePublish(activeTab)}
            disabled={publishing || !currentPost?.text}
            style={{ width: '100%', marginTop: '1.5rem', padding: '0.75rem', fontSize: '1.05rem' }}
          >
            {publishing ? 'Gözləyin...' : (scheduleTime ? 'Planla (Schedule)' : 'Dərhal Paylaş')}
          </button>

          {currentPost?.status === 'published' && <p style={{color: 'var(--success)', marginTop: '0.5rem', textAlign: 'center'}}>✓ Artıq paylaşıldı</p>}
          {currentPost?.status === 'scheduled' && <p style={{color: 'var(--accent)', marginTop: '0.5rem', textAlign: 'center'}}>⏳ Planlandı</p>}
        </div>
      </div>
    </div>
  )
}
