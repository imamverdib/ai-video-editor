import { expect, test } from '@playwright/test'

test('autosave, immediate navigation, reload, undo and export controls', async ({page, request}) => {
  const errors: string[] = []
  page.on('response', async r => { if(r.url().endsWith('/edit')) console.log('EDIT',r.status(),r.request().postData(),await r.text()) })
  page.on('console', msg => console.log('BROWSER CONSOLE:', msg.text()))
  page.on('pageerror', e => errors.push(e.message))
  const projects = await (await request.get('/api/projects')).json()
  const a = projects.find((p: {original_filename: string}) => p.original_filename === 'Browser A.mp4')
  await page.goto('/')
  await page.getByText('Browser A.mp4', {exact:true}).click()
  await expect(page.locator('.token')).toHaveCount(4)
  // Click on the left edge of the token to avoid the edit/merge buttons that sit on the right
  await page.locator('.token').nth(1).click({ position: { x: 5, y: 5 } })
  // Leave before the 450 ms debounce. JSON keepalive must persist the change.
  await page.getByRole('button', {name:'← All projects'}).click()
  await expect.poll(async () => (await (await request.get(`/api/projects/${a.id}`)).json()).edit.removed_word_ids).toEqual([1])
  await page.getByText('Browser B.mp4', {exact:true}).click()
  await expect(page.locator('.token.removed')).toHaveCount(0)
  await page.getByRole('button', {name:'← All projects'}).click()
  await page.getByText('Browser A.mp4', {exact:true}).click()
  await expect(page.locator('.token.removed')).toHaveCount(1)
  await page.reload()
  await page.getByText('Browser A.mp4', {exact:true}).click()
  await expect(page.locator('.token.removed')).toHaveCount(1)
  await page.getByRole('button', {name:'Geri al', exact:true}).click()
  await expect(page.locator('.token.removed')).toHaveCount(0)
  await expect(page.getByLabel('Export keyfiyyəti')).toBeVisible()
  expect(errors).toEqual([])
})
